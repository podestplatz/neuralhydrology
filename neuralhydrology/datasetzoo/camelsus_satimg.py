from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import concurrent.futures
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import rasterio

from neuralhydrology.datasetzoo.camelsus import CamelsUS
from neuralhydrology.utils.config import Config


class CamelsUS_SatImg(CamelsUS):
    """Data set class for the CAMELS US data set by [#]_ and [#]_ using satellite image embeddings as attributes.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
        
    References
    ----------
    .. [#] A. J. Newman, M. P. Clark, K. Sampson, A. Wood, L. E. Hay, A. Bock, R. J. Viger, D. Blodgett, 
        L. Brekke, J. R. Arnold, T. Hopson, and Q. Duan: Development of a large-sample watershed-scale 
        hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional 
        variability in hydrologic model performance. Hydrol. Earth Syst. Sci., 19, 209-223, 
        doi:10.5194/hess-19-209-2015, 2015
    .. [#] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and 
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str | None = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xr.DataArray]] = {}):
        # ensure that all embedding dimensions will be registered as static attribute
        cfg._cfg["static_attributes"] = list(range(256))

        super().__init__(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basin=basin,
            additional_features=additional_features,
            id_to_int=id_to_int,
            scaler=scaler
        )

    def _load_attributes(self) -> pd.DataFrame:
        """Override the parent method to ensure the right load_camels_us_attributes function is called."""
        num_workers = min(self.cfg.num_workers, len(self.basins))
        return load_camels_us_satimg_attributes(
            self.cfg.satimg_dir,
            basins=self.basins,
            num_workers=num_workers,
            patch_embedding_dim=self.cfg.patch_embedding_dimensions
        )


def _process_single_basin(
    basin: str,
    embeddings_dir: Path,
    raw_tile_dir: Path,
    patch_embedding_dim: Union[int, list[int], None] = None
) -> Tuple[str, Union[np.ndarray, None]]:
    """Process a single basin's embeddings.
    
    Parameters
    ----------
    basin : str
        The basin ID to process
    embeddings_dir : Path
        Directory containing the embeddings
    raw_tile_dir : Path
        Directory containing the raw tiles
        
    Returns
    -------
    Tuple[str, np.ndarray | None]
        Tuple containing the basin ID and its processed embeddings. Returns None for the embeddings if no valid data is found.
    """
    embeddings_basin_dir = embeddings_dir / basin
    if not embeddings_basin_dir.exists():
        raise FileNotFoundError(f"Satellite image directory not found for basin {basin}. Expected directory: {embeddings_basin_dir}")

    # get a list of all available embedding files
    embedding_basin_paths = list(embeddings_basin_dir.rglob('*.cropped.embed.nc'))
    
    load_class_patch_embeddings = True
    if patch_embedding_dim is not None and patch_embedding_dim != []:
        load_class_patch_embeddings = False
        if isinstance(patch_embedding_dim, list) and len(patch_embedding_dim) == 0:
            patch_embedding_dim = list(range(1024))

    # aggregate the class patch embeddings based on the overlap percentage
    aggregated_embeddings = None
    aggregated_overlap_proportion = 0
    for embedding_file in embedding_basin_paths:
        raw_tile_file = list((raw_tile_dir / basin).rglob(f"{embedding_file.name.split('.')[0]}*"))[0]

        with rasterio.open(raw_tile_file, 'r') as raw_tile:
            # percentage of area of the tile that overlaps with the basin's area
            overlap_percentage_tag: str = raw_tile.tags().get('overlap_proportion', "1.0")
            try:
                overlap_percentage: float = float(overlap_percentage_tag)
            except Exception as e:
                raise ValueError(f"Could not interpret overlap_percentage ({overlap_percentage_tag}) of raw tile file {raw_tile_file}: {e}")
            
        # load the embedding of the tile
        with xr.open_dataset(embedding_file) as embeddings:
            if load_class_patch_embeddings:
                tile_embedding: np.ndarray = embeddings["class_token"].values[0]
            else: 
                tile_embedding = embeddings.token_embeddings[:, patch_embedding_dim].mean(dim="patches")
            
        # aggregate the embedding of the tile weighted by the percentage it overlaps with the basin's area
        if aggregated_embeddings is None:
            aggregated_embeddings = tile_embedding * overlap_percentage
        else: 
            aggregated_embeddings += tile_embedding * overlap_percentage
            
        # sum the overlap_percentages to later normalize the embeddings again
        aggregated_overlap_proportion += overlap_percentage
        
    if aggregated_embeddings is None:
        print(f"No embeddings found for basin {basin}. Skipping...")
        return basin, None

    # normalize the embeddings again
    basin_embedding: np.ndarray = aggregated_embeddings / aggregated_overlap_proportion
    
    return basin, basin_embedding

def load_camels_us_satimg_attributes(
    data_dir: Path,
    basins: List[str] = [],
    num_workers: int = 1,
    patch_embedding_dim: Union[int, list[int], None] = None
) -> pd.DataFrame:
    """
    Load Satellite image embeddings as attributes
    
    Note
    ----
    The subdirectory structure of the raw tile dir and the embeddings dir are assumed to be the same:
    <basin_id>/<utm_tile_name>/<timestamp>_<utm_tile_name>.cropped{.tif|.embeddings.nc}.
    """
    def init_basin_attributes(basin_attribute_columns: List[int]) -> pd.DataFrame:
        basin_attributes = pd.DataFrame({}, columns=['basin'] + basin_attribute_columns)
        basin_attributes.set_index(keys=["basin"], inplace=True)
        return basin_attributes
    
    raw_tile_dirname = "utm_tile_based"
    embeddings_dirname = "utm_tile_based_embedded_ae"

    embeddings_dir = data_dir / embeddings_dirname
    raw_tile_dir = data_dir / raw_tile_dirname

    # the dataframe housing the embeddings as attributes (each embedding dimension corresponds to one attribute column)
    basin_attributes = None
    basin_attribute_columns = None

    if not basins:
        basins = [d.name for d in embeddings_dir.glob("*") if d.is_dir()]

    # Create a partial function with the fixed arguments
    process_func = partial(
        _process_single_basin, 
         embeddings_dir=embeddings_dir,
         raw_tile_dir=raw_tile_dir,
         patch_embedding_dim=patch_embedding_dim
    )

    # Process basins in parallel
    if num_workers > 1:     
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_func, basin) for basin in basins]
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(basins), 
                             desc="Processing basins"):
                basin, embeddings = future.result()
                
                # lazily initialise basin attributes dataframe with the actual number of columns
                if basin_attributes is None:
                    basin_attribute_columns = list(range(embeddings.shape[0]))
                    basin_attributes = init_basin_attributes(basin_attribute_columns)

                if embeddings is not None:
                    basin_attributes.loc[basin, basin_attribute_columns] = embeddings
                    
    else:
        for basin in tqdm(basins, desc="Processing basins"):
            basin, embeddings = process_func(basin)

            # lazily initialise basin attributes dataframe with the actual number of columns
            if basin_attributes is None:
                basin_attribute_columns = list(range(embeddings.shape[0]))
                basin_attributes = init_basin_attributes(basin_attribute_columns)

            if embeddings is not None:
                basin_attributes.loc[basin, basin_attribute_columns] = embeddings
            
    return basin_attributes 


if __name__ == "__main__":
    df = load_camels_us_satimg_attributes(
        Path("/system/user/publicdata/hydrology/CAMELS_US_Satellite_Images/"),
        ["01013500", "01030500"],
        num_workers=2,
        patch_embedding_dim=[]
    )
    print(df)