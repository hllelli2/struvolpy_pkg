__author__ = "Luc Elliott"
__date__ = "27 Jun 2023"
__version__ = "1.0.1"

from pathlib import Path
import numpy as np
import logging
from copy import copy
from datetime import datetime

import mrcfile
from typing import Sequence
from .Structure import Structure
from struvolpy.wrappers import requires_TEMPy

try:
    from TEMPy.maps.em_map import Map
    from TEMPy.maps.map_parser import MapParser
    from TEMPy.protein.structure_blurrer import StructureBlurrer
    from TEMPy.protein.scoring_functions import ScoringFunctions
except ImportError:
    pass


class VolumeParser(object):
    """
    A class for parsing volume data from an MRC file.

    Args:
        filename (str): The path to the MRC file to be parsed.
        exists (bool): Whether the file exists or not. Defaults to True.

    """

    def __init__(self, filename, exists=True) -> None:
        self.filename = filename

        if exists:
            self.mrc_data = mrcfile.open(filename)
            self.read_data()

        self.read_header(exists=exists)
        self.alter_grid()

        if exists:
            self.close()

    """Public methods"""

    def read_header(self, exists: bool = True) -> None:
        """
        Reads the header information from the MRC file.

        If the file exists, the header information is read from the MRC file object.
        If the file does not exist, a temporary MRC object is created and its default attributes are copied.

        Args:
            exists (bool): Whether the file exists or not. Defaults to True.

        Returns:
            None
        """
        if not exists:
            tmpmrc = mrcfile.mrcobject.MrcObject()
            tmpmrc._create_default_attributes()
            self.header = tmpmrc.header.copy()
        else:
            self.header = self.mrc_data.header.copy()

    def read_data(self) -> None:
        """
        Reads the volume data from the MRC file.

        The volume data is stored as a numpy array, as the type float64,
        in the 'data' attribute of the VolumeParser object.

        Returns:
            None
        """
        self.data = self.mrc_data.data.astype(np.float64)
        self.voxel_spacing = self.mrc_data.voxel_size.item()[0]

    def close(self) -> None:
        """
        Closes the MRC file object.

        Returns:
            None
        """
        self.mrc_data.close()

    def origin(self) -> Sequence[float]:
        """
        Returns the origin of the Volume object.

        If the origin is zero, the function uses the nxstart values instead.
        The origin is then converted to physical units using the voxel spacing.

        Returns:
            tuple: A tuple containing the origin of the Volume object in the x, y, and z directions.
        """
        if all(x == 0.0 for x in self.header.origin.item()):
            logging.warning("Origin is zero, use nxstart values instead.")
            self.header.origin = (
                self.header.nxstart * self.voxel_spacing,
                self.header.nystart * self.voxel_spacing,
                self.header.nzstart * self.voxel_spacing,
            )

        return self.header.origin

    def nstart(self):
        """
        Returns the starting indices of the Volume object.

        Returns:
            tuple: A tuple containing the starting indices of the Volume object in the x, y, and z directions.
        """
        return self.header.nxstart, self.header.nystart, self.header.nzstart

    def grid(self):
        """
        Returns the grid of the Volume object.

        Returns:
            numpy.ndarray: The grid of the Volume object.
        """
        return self.data

    def alter_grid(self):
        """
        Alters the grid of the Volume object according to the mapc, mapr, and maps values in the header.

        Swaps the axes of the data array if necessary to match the mapc, mapr, and maps values in the header.
        Updates the mapc, mapr, and maps values in the header to match the new axis order.

        Returns:
            None
        """

        axis_cols = {
            1: "mapc",
            2: "mapr",
            3: "maps",
        }

        if self.header.mapc != 1:
            self.data = np.swapaxes(self.data, 2, self.header.mapc - 1)
            self.header[axis_cols[self.header.mapc]] = self.header.mapc
            self.header.mapc = 1

        if self.header.mapr != 2:
            self.data = np.swapaxes(self.data, 1, self.header.mapr - 1)
            self.header.mapr = 2
            self.header.maps = 3


class Volume(object):
    @classmethod
    def from_file(cls, filename):
        """
        Creates a Volume object from a file.

        Args:
            filename (str or Path): The path to the file.

        Returns:
            Volume: The Volume object.

        Raises:
            NotImplementedError: If the file extension is not supported.
            ValueError: If the file extension is not supported.

        """
        try:
            filename = filename.name
        except AttributeError:
            filename = filename

        if filename.endswith(("ccp4", "map", "mrc")):
            pass
        elif filename.endswith(("xplor", "cns")):
            raise NotImplementedError("XPLOR and CNS file formats are not supported.")
        else:
            raise ValueError("Extension of file is not supported.")

        return cls(VolumeParser(filename))

    @classmethod
    def from_data(
        cls, grid, voxelspacing, origin, filename_output="VolumeFromData.mrc"
    ):
        """
        Creates a Volume object from data.

        Args:
            grid (numpy.ndarray): The grid of the Volume object.
            voxelspacing (float): The voxel spacing of the Volume object.
            origin (tuple): The origin of the Volume object.
            filename_output (str or Path): The path to the output file. Defaults to "VolumeFromData.mrc".

        Returns:
            Volume: The Volume object.

        """
        volume = VolumeParser(filename_output, exists=False)
        volume.data = grid
        volume.voxel_spacing = voxelspacing
        volume.header.origin = tuple(origin)

        volume.header.nx = volume.header.mx = grid.shape[2]
        volume.header.ny = volume.header.my = grid.shape[1]
        volume.header.nz = volume.header.mz = grid.shape[0]

        volume.header.cella = (
            volume.header.mx * voxelspacing,
            volume.header.my * voxelspacing,
            volume.header.mz * voxelspacing,
        )

        return cls(volume)

    @classmethod
    def from_TEMPy_map(cls, tempy_map, filename_output="VolumeFromTempy.mrc"):
        print(type(tempy_map))
        grid = tempy_map.fullMap

        voxelspacing = tempy_map.apix[0]
        origin = tempy_map.origin
        return cls.from_data(grid, voxelspacing, origin, filename_output)

    def __init__(self, volume_parser) -> None:
        """
        Initializes a Volume object.

        Args:
            volume_parser (VolumeParser): The VolumeParser object.
        """

        self.__filename = volume_parser.filename
        self.__header = volume_parser.header
        vx = vy = vz = volume_parser.voxel_spacing
        self.__voxel_size = vx, vy, vz
        self.__origin = volume_parser.origin()
        self.__grid = volume_parser.grid()
        self.__nstart = volume_parser.nstart()
        self.__simulated = False
        self.__resolution = -1.0
        self.__tempy_map = None
        self.__threshold = -1.0

    """Properties"""

    @property
    def resolution(self) -> float:
        """
        Gets the resolution of the Volume object.

        Returns:
            float: The resolution.
        """
        return self.__resolution

    @resolution.setter
    def resolution(self, resolution: float) -> None:
        """
        Sets the resolution of the Volume object.

        Args:
            resolution (float): The resolution.
        """
        # try float conversion
        self.__resolution = resolution

    @property
    def simulated(self):
        """
        Gets the simulated property of the Volume object.

        Returns:
            bool: The simulated property.
        """
        return self.__simulated

    @simulated.setter
    def simulated(self, simulated: bool) -> None:
        """
        Sets the simulated property of the Volume object.

        Args:
            simulated (bool): The simulated property.

        Raises:
            TypeError: If the simulated property is not a boolean.
        """
        if not isinstance(simulated, bool):
            raise TypeError("Must be a boolean")
        self.__simulated = simulated

    @property
    def grid(self) -> np.ndarray:
        """
        Gets the grid of the Volume object.

        Returns:
            np.ndarray: The grid.
        """
        return self.__grid

    @grid.setter
    def grid(self, grid) -> None:
        """
        Sets the grid of the Volume object.

        Args:
            grid (np.ndarray): The grid.
        """
        # add check to make sure grid is a numpy array
        if not isinstance(grid, np.ndarray):
            raise TypeError("grid is not a numpy array")
        if len(grid.shape) != 3:
            raise ValueError("grid must be a 3D numpy array")

        self.__grid = grid

    @property
    def voxelsize(self) -> np.ndarray:
        """
        Gets the voxel size of the Volume object.

        Returns:
            np.ndarray: The voxel size.
        """

        return np.asarray([x for x in self.__voxel_size])

    @voxelsize.setter
    def voxelsize(self, voxelsizes) -> None:
        """
        Sets the voxel size of the Volume object.

        Args:
            voxelsizes (list): The voxel size as a list of 3 floats.
        """
        if len(voxelsizes) != 3:
            raise TypeError("voxelsize must be a list of 3 floats")
        try:
            voxelsizes = tuple(voxelsizes)
        except TypeError:
            raise TypeError("voxelsize must be a list of 3 floats")

        vx, vy, vz = voxelsizes
        self.__voxel_size = vx, vy, vz

    @property
    def origin(self) -> tuple:
        """
        Gets the origin of the Volume object.

        Returns:
            tuple: The origin.
        """
        return self.__origin.item()

    @origin.setter
    def origin(self, origin) -> None:
        """
        Sets the origin of the Volume object.

        Args:
            origin (list): The origin as a list of 3 floats or integers.
        """
        if len(origin) != 3:
            raise TypeError("origin must be a list of 3 floats or integers")
        # ox, oy, oz = origin
        self.__origin = np.array(origin)

    @property
    def start(self) -> np.ndarray:
        """
        Gets the start of the Volume object.

        Returns:
            np.ndarray: The start.
        """

        return np.array([o / self.voxelspacing for o in self.__origin.item()])

    @start.setter
    def start(self, start: Sequence) -> None:
        """
        Sets the start of the Volume object.

        Args:
            start (tuple or list): The start as a list of 3 floats or
            integers.
        """
        ox, oy, oz = [x * self.voxelspacing for x in start]
        self.origin = ox, oy, oz

    @property
    def threshold(self) -> float:
        """
        Gets the threshold of the Volume object.

        Returns:
            float: The threshold.
        """
        if self.__threshold == -1.0:
            self.calculate_threshold()
        return self.__threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        """
        Sets the threshold of the Volume object.

        Args:
            threshold (float): The threshold.

        Raises:
            TypeError: If threshold is not a float.
        """
        if not isinstance(threshold, float):
            raise TypeError("threshold must be a float")

        self.__threshold = threshold

    @property
    def header(self) -> dict:
        """
        Gets the header of the Volume object.

        Returns:
            dict: The header.
        """
        return self.__header

    @property
    def voxelspacing(self) -> float:
        """
        Gets the voxel spacing of the Volume object.

        Returns:
            float: The voxel spacing.
        """
        return self.__voxel_size[0]

    @property
    def shape(self, format="zyx") -> tuple:
        """
        Gets the shape of the Volume object.

        Args:
            format (str): The format of the shape. Either "xyz" or "zyx".

        Returns:
            tuple: The shape.

        Raises:
            ValueError: If format is not "xyz" or "zyx".
        """
        if format == "xyz":
            return self.__grid.shape[::-1]
        elif format == "zyx":
            return self.__grid.shape
        else:
            raise ValueError("format must be either xyz or zyx")

    @property
    def dimensions(self) -> np.ndarray:
        """
        Gets the dimensions of the Volume object.

        Returns:
            np.ndarray: The dimensions.
        """
        return np.asarray([x * self.voxelspacing for x in self.shape][::-1])

    @property
    def nx(self) -> int:
        """
        Gets the nx of the Volume object.

        Returns:
            int: The nx.
        """
        return self.shape[2]

    @property
    def ny(self) -> int:
        """
        Gets the ny of the Volume object.

        Returns:
            int: The ny.
        """
        return self.shape[1]

    @property
    def nz(self) -> int:
        """
        Gets the nz of the Volume object.

        Returns:
            int: The nz.
        """
        return self.shape[0]

    @property
    def dmin(self) -> float:
        """
        Gets the dmin of the Volume object.

        Returns:
            float: The dmin.
        """
        return np.min(self.__grid.astype(np.float32))

    @property
    def dmax(self) -> float:
        """
        Gets the dmax of the Volume object.

        Returns:
            float: The dmax.
        """
        return np.max(self.__grid.astype(np.float32))

    @property
    def dmean(self) -> float:
        """
        Gets the dmean of the Volume object.

        Returns:
            float: The dmean.
        """
        return np.mean(self.__grid.astype(np.float32))

    @property
    def rms(self) -> float:
        """
        Gets the rms of the Volume object.

        Returns:
            float: The rms.
        """
        return np.sqrt(np.mean(np.square(self.__grid.astype(np.float32) - self.dmean)))

    @property
    def filepathway(self) -> str:
        """
        Gets the absolute filepathway of the Volume object.

        Returns:
            str: The filepathway.
        """
        return str(Path(self.__filename).resolve())

    @property
    def TEMPy_map(self) -> "Map":
        """
        Gets the TEMPy map of the Volume object.

        Returns:
            TEMPy.MapParser: The TEMPy map.
        """
        if self.__tempy_map is None:
            self.create_TEMPy_object()
        return self.__tempy_map

    @property
    def filename(self) -> str:
        """
        Gets the filename of the Volume object.

        Returns:
            str: The filename.
        """
        return Path(self.__filename).name

    @filename.setter
    def filename(self, filename) -> None:
        """
        Sets the filename of the Volume object.

        Args:
            filename (str): The filename.
        """
        self.__filename = filename

    """Public Methods"""

    @requires_TEMPy
    def calculate_threshold(self, simulated=False):
        """
        Calculates the threshold for the map.

        Args:
            simulated (bool): Whether to use code similar to CCPEM TEMPy Scores_process.py pipeline.

        Raises:
            ValueError: If no resolution is specified.

        Returns:
            None
        """
        if self.__tempy_map is None:
            self.create_tempy_object()

        if simulated:
            if not self.__resolution:
                raise ValueError("No resolution specified")

            if self.__resolution > 10.0:
                t = 2.5
            elif self.__resolution > 6.0:
                t = 2.0
            else:
                t = 1.5
            threshold = t * self.grid.std()
        else:
            threshold = ScoringFunctions().calculate_map_threshold(self.__tempy_map)

        self.__threshold = threshold

    @requires_TEMPy
    def create_TEMPy_object(self) -> None:
        """
        Creates a TEMPy Map object.

        Returns:
            None
        """

        self.__tempy_map = Map(
            self.__grid,
            self.__origin.item(),
            self.__voxel_size,
            self.__filename,
            MapParser.parseCcpemHeader(self.__header),
        )

    def duplicate(self) -> "Volume":
        """
        Duplicates the Volume object.

        Returns:
            Volume: The duplicate Volume object.
        """
        duplicated_volume = copy(self)
        if self.__tempy_map is not None:
            duplicated_volume.__tempy_map = copy(self.__tempy_map)
        return duplicated_volume

    def to_file(self, filename=None, output_format=None, overwrite=False):
        """
        Writes the Volume object to a file.

        Args:
            filename (str): The filename to write to.
            output_format (str): The output format.
            overwrite (bool): Whether to overwrite the file if it already exists.

        Raises:
            NotImplementedError: If the output format is not supported.
        """
        if filename is None:
            filename = self.filepathway
        if output_format is not None:
            filename = str(Path(filename).with_suffix("." + output_format))

        output_format = Path(filename).suffix[1:]
        if not filename.endswith(("ccp4", "map", "mrc")):
            raise NotImplementedError("Only MRC format is supported at this time.")

        with mrcfile.new(filename, overwrite=True) as mrc:
            mrc.set_data(self.__grid.astype(np.float32))
            mrc.header.cella = tuple(self.dimensions)

            mrc.voxel_size = self.__voxel_size
            mrc.header.origin = self.__origin

            mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = self.start
            mrc.update_header_from_data()

            mrc.header.dmin = self.dmin
            mrc.header.dmax = self.dmax
            mrc.header.dmean = self.dmean
            mrc.header.rms = self.rms

            now = datetime.now()
            date_time = now.strftime("%H:%M %d/%m/%y")
            mrc.header.label[0] = f"{output_format.upper()} created at " + str(
                date_time
            )

        with mrcfile.open(filename) as mrc:
            if not mrc.validate():
                raise ValueError("MRC file is not valid")


class SimulatedMap:
    """
    A class for simulating a map from a volume and a structure.
    Class set up for future development.

    Attributes:
        None

    Methods:
        simulate_map(volume: Volume, structure: Structure, resolution: float = -1.0, normalise: bool = False) -> Volume:
            Simulates a map from a volume and a structure.


    """

    @requires_TEMPy
    def __init__(self) -> None:
        pass

    @staticmethod
    def simulate_map(
        volume: Volume,
        structure: Structure,
        resolution: float = -1.0,
        normalise: bool = False,
    ) -> Volume:
        """
        Simulates a map from a volume and a structure.

        Args:
            volume (Volume): The volume to use for the simulation.
            structure (Structure): The structure to use for the simulation.
            resolution (float, optional): The resolution to use for the simulation. Defaults to -1.0.
            normalise (bool, optional): Whether to normalise the simulated map. Defaults to False.

        Returns:
            Volume: The simulated map.
        """
        if resolution == -1.0:
            resolution = volume.resolution
        if resolution is None:
            raise AssertionError(
                "resolution must be specified "
                "either in the Volume class or as a kwarg"
            )

        sb = StructureBlurrer(with_vc=True)
        simulatedmap_tempy = sb._gaussian_blur_real_space_vc(
            structure.to_TEMPy(),
            resolution,
            volume.TEMPy_map,
        )

        if normalise:
            simulatedmap_tempy.normalise()

        simulatedmap_volume = Volume.from_TEMPy_map(simulatedmap_tempy)
        simulatedmap_volume.resolution = resolution
        simulatedmap_volume.calculate_threshold(simulated=True)
        simulatedmap_volume.simulated = True

        return simulatedmap_volume


"""class _StructureBlurrerbfac(StructureBlurrer):
    
    Not currently used.
    

    def __init__(self, outname: str, with_vc=False):
        self.outname = outname
        super().__init__(with_vc=with_vc)

    def _gaussian_blur_real_space_vc_bfac(
        self,
        struct,
        resolution,
        exp_map,
        SIGMA_COEFF=0.356,
        cutoff=4.0,
    ):
        if not self.use_vc:
            return None

        import voxcov as vc

        blur_vc = vc.BlurMap(
            exp_map.apix,
            exp_map.origin,
            [exp_map.x_size(), exp_map.y_size(), exp_map.z_size()],
            cutoff,
        )

        # Constant for the b-factor sigma conversion
        SIGMA_CONV = 3 / (8 * (np.pi**2))

        for a in struct.atomList:
            sigma = SIGMA_CONV * a.temp_fac * resolution * SIGMA_COEFF
            height = 0.4 / sigma

            blur_vc.add_gaussian(
                [a.x, a.y, a.z],
                a.get_mass() * height,  # height
                SIGMA_COEFF * resolution,  # width
            )
        full_map = blur_vc.to_numpy()

        return Map(
            full_map,
            exp_map.origin,
            exp_map.apix,
            self.outname,
        )
"""
