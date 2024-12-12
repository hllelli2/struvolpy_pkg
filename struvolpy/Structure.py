__author__ = "Luc Elliott"
__date__ = "14 Jun 2023"
__version__ = "1.0.7"

import numpy as np
from pathlib import Path
from copy import copy
import gemmi
import logging
from typing import Dict, List, Union
from struvolpy.wrappers import requires_TEMPy
import warnings

try:
    from TEMPy.protein.structure_parser import mmCIFParser
except ImportError:
    pass


FILE_EXTENSIONS = ("pdb", "mmcif", "mmjson", "cif")

logger = logging.getLogger(__name__)


class Structure(object):
    """
    Represents a protein structure parsed from a PDB/mmCIF file.

    """

    @classmethod
    def from_file(cls, filename: str, hetatm: bool = False, water: bool = False):
        """
        Initializes a Structure object from a PDB/mmCIF file.

        Args:
            filename (str): The path to the PDB/mmCIF file.
            hetatm (bool, optional): Whether to include HETATM records in the
            structure. Defaults to False.
            water (bool, optional): Whether to include water molecules in the
            structure. Defaults to False.

        Returns:
            Structure: A Structure object representing the parsed protein
            structure.

        """

        filename = str(filename)

        if not filename.endswith(FILE_EXTENSIONS):
            warnings.warn(
                f"File extension is not {', '.join(FILE_EXTENSIONS[:-1])} or {FILE_EXTENSIONS[-1]}"
                " and may not be parsed correctly"
            )
        gemmi_structure = gemmi.read_structure(filename)
        gemmi_structure.setup_entities()

        if not hetatm:
            try:
                gemmi_structure.remove_ligands_and_waters()
            except RuntimeError as e:
                logger.info("Removed ligands and waters failed, continuing")
                pass
        if not water:
            gemmi_structure.remove_waters()

        gemmi_structure.remove_empty_chains()

        chain_id = 65
        for chain in gemmi_structure[0]:
            if chain.name == "":
                chain.name = chr(chain_id)
                chain_id += 1

        return cls(filename, gemmi_structure)

    @classmethod
    def from_gemmi(
        cls,
        gemmi_structure: gemmi.Structure,
        filename: Union[str, Path] = "",
        hetatm: bool = False,
        water: bool = False,
    ) -> "Structure":
        """
        Initializes a Structure object from a gemmi.Structure object.

        Args:
            gemmi_structure (gemmi.Structure): The gemmi.Structure object to initialize the Structure from.
            filename (str or Path, optional): The name of the file the structure was parsed from. Defaults to None.
            hetatm (bool, optional): Whether to include HETATM records in the structure. Defaults to False.
            water (bool, optional): Whether to include water molecules in the structure. Defaults to False.

        Returns:
            Structure: A Structure object representing the parsed protein structure.

        Raises:
            AssertionError: If the input is not a gemmi.Structure object.
        """

        if not isinstance(gemmi_structure, gemmi.Structure):
            AssertionError("Not a gemmi structure cannot use from_gemmi")

        if not hetatm:
            try:
                gemmi_structure.remove_ligands_and_waters()
            except RuntimeError:
                logger.debug("Removed ligands and waters failed")
                pass
        if not water:
            gemmi_structure.remove_waters()

        gemmi_structure.remove_empty_chains()

        gemmi_structure.setup_entities()
        return cls(filename, gemmi_structure)

    def __init__(self, filename: Union[str, Path], structure: gemmi.Structure) -> None:
        """
        Initializes a Structure object from a file name and a gemmi.Structure object.

        Args:
            filename (str or Path): The name of the file the structure was parsed from.
            structure (gemmi.Structure): The gemmi.Structure object to initialize the Structure from.

        Returns:
            None

        Raises:
            None
        """
        self.__gemmi_structure = structure
        self.__gemmi_structure.setup_entities()
        self.__gemmi_structure.assign_label_seq_id()
        self.__tree = None
        self.__gemmi_structure.name = str(filename)
        self.__filename = filename

    """Properties"""

    @property
    def filepathway(self) -> str:
        """
        Gets the absolute filepathway of the Structure object.

        Returns:
            str: The filepathway.
        """
        return str(Path(self.__filename).resolve())

    @property
    def structure(self) -> gemmi.Structure:
        """
        Returns the gemmi.Structure object representing the protein structure.

        Returns:
            gemmi.Structure: The gemmi.Structure object representing the
            protein structure.
        """
        return self.__gemmi_structure

    @structure.setter
    def structure(self, new_structure: gemmi.Structure) -> None:
        """
        Sets the gemmi.Structure object representing the protein structure.

        Args:
            new_structure (gemmi.Structure): The gemmi.Structure object
            representing the protein structure.

        Returns:
            None
        """
        self.__gemmi_structure = new_structure

    @property
    def atoms(self) -> List[gemmi.Atom]:
        """
        Returns a list of all atoms in the protein structure.

        Returns:
            List[gemmi.Atom]: A list of all atoms in the protein structure.
        """
        return [atom for atom in self.__get_atoms()]

    @property
    def alpha_carbons(self) -> List[gemmi.Atom]:
        """
        Returns a list of all alpha-carbon atoms in the protein structure.

        Returns:
            List[gemmi.Atom]: A list of all alpha-carbon atoms in the protein structure.
        """
        return [atom for atom in self.__get_atoms() if atom.name == "CA"]

    @property
    def chains(self) -> List[gemmi.Chain]:
        """
        Returns a list of all chains in the protein structure.

        Returns:
            List[gemmi.Chain]: A list of all chains in the protein structure.
        """
        return [chain for chain in self.structure[0]]

    @property
    def filename(self) -> str:
        """
        Returns the name of the file from which the structure was loaded.

        Returns:
            str: The name of the file.
        """
        return self.__gemmi_structure.name

    @filename.setter
    def filename(self, filename: str) -> None:
        """
        Sets the name of the file from which the structure was loaded.

        Args:
            filename (str): The name of the file.

        Returns:
            None
        """
        self.__gemmi_structure.name = str(filename)

    @property
    def coor(self) -> np.ndarray:
        """
        Returns the coordinates of all atoms in the protein structure.

        Returns:
            np.ndarray: A numpy array of shape (n_atoms, 3) containing
            the x, y, and z coordinates of all atoms.
        """
        return np.asarray([*zip(*[atom.pos for atom in self.__get_atoms()])])

    @coor.setter
    def coor(self, coor: np.ndarray) -> None:
        """
        Sets the coordinates of all atoms in the protein structure.

        Args:
            coor (np.ndarray): A numpy array of shape (n_atoms, 3)
            containing the new x, y, and z coordinates of all atoms.

        Returns:
            None
        """
        assert coor.shape == self.coor.shape
        for atom, pos in zip(self.__get_atoms(), coor.T):
            atom.pos = gemmi.Position(*pos)

    @property
    def alpha_carbon_coor(self) -> np.ndarray:
        """
        Returns the coordinates of all alpha-carbon atoms in the protein structure.

        Returns:
            np.ndarray: A numpy array of shape (n_atoms, 3) containing
            the x, y, and z coordinates of all alpha-carbon atoms.
        """
        return np.asarray(
            [
                *zip(
                    *[
                        atom.pos
                        for atom in self.__get_atoms()
                        if atom.name == "CA" and atom.element.name == "C"
                    ]
                )
            ]
        )

    @property
    def bfacs(self) -> np.ndarray:
        """
        Returns the values stored in the b-factor column of all atoms in
          the protein structure.

        Returns:
            np.ndarray: A numpy array of shape (n_atoms,) containing the
            (typically) b-factor values of all atoms.
        """
        return np.asarray([atom.b_iso for atom in self.__get_atoms()])

    @bfacs.setter
    def bfacs(self, new_bfacs: np.ndarray) -> None:
        """
        Sets the values stored in the b-factor column of all atoms
        in the protein structure.

        Args:
            bfacs (np.ndarray): A numpy array of shape (n_atoms,)
            containing the new (typically) b-factor values of all atoms.

        Returns:
            None
        """
        assert new_bfacs.shape == self.bfacs.shape
        for atom, bfac in zip(self.__get_atoms(), new_bfacs):
            atom.b_iso = bfac

    @property
    def weights(self) -> np.ndarray:
        """
        Returns the atomic weights of all atoms in the protein structure.

        Returns:
            np.ndarray: A numpy array of shape (n_atoms,)
            containing the atomic weights of all atoms.
        """
        return np.asarray([atom.element.weight for atom in self.__get_atoms()])

    @property
    def centre_of_mass(self) -> np.ndarray:
        """
        Calculates the centre of mass of the protein structure.

        Returns:
            np.ndarray: A numpy array of shape (3,) containing the x,
            y, and z coordinates of the centre of mass.
        """
        return np.average(self.coor, axis=1, weights=self.weights)

    @property
    def sequence(self):
        """
        Returns the sequence of the protein structure.

        The sequence is defined as the one-letter code of the element
        name of each atom in the protein structure.

        Returns:
            np.ndarray: A numpy array of shape (n_atoms,) containing
            the one-letter code of the element name of each atom.
        """
        return np.asarray([atom.element.name for atom in self.__get_atoms()])

    @property
    def residues(self):
        """
        Returns a list of all residues in the protein structure.

        Returns:
            List[gemmi.Residue]: A list of all residues in the protein structure.
        """
        return [residue for residue in self.__get_residues()]

    """Private Methods"""

    def __get_atoms(self):
        """
        A generator that yields all atoms in the protein structure.

        Yields:
            gemmi.Atom: The next atom in the protein structure.
        """
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        yield atom

    def __get_residues(self):
        """
        A generator that yields all residues in the structure.

        Note:
            Not to be mistaken with the get_residues method which
            returns a dictionary of chain names as the key and the one-letter
            residue codes as the value.

        Yields:
            gemmi.Residue: The next residue in the protein structure.
        """
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    yield residue

    def __build_tree(self):
        """
        Builds a Ball tree from the coordinates of the atoms in the
        protein structure.

        Uses the BallTree implementation from the sklearn.neighbors module.

        The tree is stored as a private attribute of the Structure object.

        Returns:
            None
        """
        from sklearn.neighbors import BallTree

        self.__tree_coords = np.asarray(
            np.asarray(
                [
                    self.alpha_carbon_coor[0],
                    self.alpha_carbon_coor[1],
                    self.alpha_carbon_coor[2],
                ]
            )
        ).T
        self.__tree = BallTree(self.__tree_coords)

    """Public Methods"""

    def add_model(self, model: gemmi.Model):
        """
        Adds a new model to the Structure object.

        Args:
            model (gemmi.Model): The model to be added.

        Returns:
            None
        """
        self.__gemmi_structure.add_model(model)

    def add_chain(self, chain: gemmi.Chain, model_idx: int = 0, **kwargs):
        """
        Adds a new chain to the specified model of the Structure object.

        Args:
            chain (gemmi.Chain): The chain to be added.
            model_idx (int): The index of the model to which the
            chain will be added. Default is 0.
            **kwargs: Additional keyword arguments.
                pos (int): The position at which to insert the chain.
                Default is -1 (append to the end).
                unique_name (bool): Whether to ensure that the chain
                name is unique. Default is True.

        Returns:
            None
        """

        pos = kwargs.get("pos", -1)
        unique_name = kwargs.get("unique_name", True)
        self.__gemmi_structure[model_idx].add_chain(
            chain, pos=pos, unique_name=unique_name
        )

    def translate(self, vector):
        """
        Translate the coordinates of the atoms in the structure by a
        given vector.

        Args:
            vector (np.ndarray or tuple or list): A 3D numpy array or tuple or list
            representing the vector by which to translate the coordinates.

        Returns:
            None

        Raises:
            AssertionError: If the shape of the vector is not (3,).
        """
        vector = np.array(vector)
        assert vector.shape == (3,)
        x_translate, y_translate, z_translate = vector
        for atom in self.atoms:
            atom.pos.x += x_translate
            atom.pos.y += y_translate
            atom.pos.z += z_translate

    def rotate(self, rotation_matrix: np.ndarray):
        """
        Rotate the coordinates of the atoms in the structure
        using a rotation matrix.

        Args:
            rotation_matrix (np.ndarray): A 3x3 numpy array
            representing the rotation matrix.

        Returns:
            None

        Raises:
            AssertionError: If the shape of the rotation matrix is not (3, 3).
        """
        assert rotation_matrix.shape == (3, 3)
        centreofmass = self.centre_of_mass.copy()
        self.translate(-centreofmass)
        self.coor = np.dot(rotation_matrix, self.coor)
        rotated_centreofmass = np.dot(rotation_matrix, centreofmass)
        self.translate(rotated_centreofmass)

    # Structure comparisons

    def overlap(
        self,
        structure,
        collision_threshold=2,
        clashes_threshold=0.05,
    ):
        """
        Determines whether two structures overlap by checking if they have a minimum number of overlapping points. Check carbon alpha atoms only.

        Args:
            structure (Structure): The structure to compare with.
            collision_threshold (float): The maximum distance between two points for them to be considered colliding.
            clashes_threshold (float): The minimum fraction of overlapping points required for the structures to be considered overlapping.

        Returns:
            bool: True if the structures overlap, False otherwise.

        Raises:
            None
        """

        if self.__tree is None:
            self.__build_tree()

        coords2 = np.asarray(
            np.asarray(
                [
                    structure.alpha_carbon_coor[0],
                    structure.alpha_carbon_coor[1],
                    structure.alpha_carbon_coor[2],
                ]
            )
        ).T

        if (np.max(self.__tree_coords, axis=0) < np.min(coords2, axis=0)).any() or (
            np.max(coords2, axis=0) < np.min(self.__tree_coords, axis=0)
        ).any():
            return False

        distances, _ = self.__tree.query(coords2)
        clashes_threshold *= min(len(self.__tree_coords), len(coords2))

        if int(sum(distances < collision_threshold)[0]) > clashes_threshold:
            return True
        else:
            return False

    def get_coordinate_of_residue(self, residue_number, atom_name="CA"):
        """
        Returns the coordinates of the specified atom in the specified residue.

        Args:
            residue_number (int): The number of the residue to search for.
            atom_name (str): The name of the atom to search for. Defaults to "CA".

        Returns:
            numpy.ndarray: The coordinates of the specified atom in the specified residue.

        """
        for residue in self.__get_residues():
            if residue.seqid.num == residue_number:
                for atom in residue:
                    if atom.name == atom_name:
                        return np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                else:
                    raise ValueError(
                        f"Atom {atom_name} not found in residue {residue.seqid.num}"
                    )
        else:
            raise ValueError(f"Residue {residue_number} not found")

    def rmsd(self, structure):
        """
        Calculates the root-mean-square deviation (RMSD) between
        two structures.

        Args:
            structure (Structure): The structure to compare with.

        Returns:
            float: The RMSD value (\u212B).

        Raises:
            AssertionError: If the number of coordinates in the two
            structures is not equal.
        """
        assert len(self.coor) == len(structure.coor)
        return np.sqrt(((self.coor - structure.coor) ** 2).mean() * 3)

    # Structure operations

    def duplicate(self):
        """
        Returns a deep copy of the current structure object.

        Returns:
            Structure: A new instance of the Structure class with the
            same attributes as the current object.

        Example:
            >>> new_structure = structure.duplicate()
        """
        new_structure = copy(self)
        new_structure.structure = self.structure.clone()
        return new_structure

    def combine(self, structure):
        """
        Combines the current structure with another structure.

        Args:
            structure (Structure): The structure to combine with.

        Returns:
            Structure: A new instance of the Structure class with the
            combined structure.

        Example:
            >>> combined_structure = structure1.combine(structure2)
        """
        combined_structure = self.duplicate()

        for model in structure.structure:
            for chain in model:
                combined_structure.structure[0].add_chain(
                    chain, pos=-1, unique_name=True
                )

        return combined_structure

    def get_residues(self, model_idx: int = 0) -> Dict[str, str]:
        """
        Returns a dictionary of chain names as the key and the one-letter
        residue codes as the value.

        Args:
            model_idx (int): The index of the model to which the chain will be
            added. Default is 0.

        Returns:
            A dictionary with chain names as the keys and one-letter residue
            codes as the values.


        Author: Adam Simpkin
        """
        chain2data = {}
        unique_chains = []
        for chain in self.__gemmi_structure[model_idx]:
            id = chain.name
            seq = ""
            for residue in chain.first_conformer():  # Only use the main conformer
                res_info = gemmi.find_tabulated_residue(residue.name)
                if res_info.is_amino_acid():  # Only amino acids
                    seq += res_info.one_letter_code
            if seq not in unique_chains:
                chain2data[id] = seq
                unique_chains.append(seq)

        return chain2data

    def to_fasta(self, seqout="", model_idx: int = 0):
        """Function to get sequence from the Structure.

        Args:
            seqout (str): The path to the output sequence file.
            model_idx (int): The index of the model to which the chain will
            be added. Default is 0.
        Returns:
            None

        Raises:
            None

        Author: Adam Simpkin
        """

        chain2data = self.get_residues(model_idx=model_idx)

        if not seqout:
            seqout = f"{Path(self.filename).stem}.fasta"

        with open(seqout, "w") as f_out:
            for k, v in chain2data.items():
                if v != "":
                    f_out.write(f">{k}\n{v}\n")

    def split_into_chains(self):
        """Splits the structure into separate structures, one for each chain.

        Returns:
            A list of Structure objects, one for each chain in the
            original structure.

        Raises:
            None

        """
        structures = []

        for chain in self.chains:
            new_structure = self.duplicate()
            new_structure.structure = gemmi.Structure()
            new_structure.add_model(gemmi.Model(1))
            new_structure.add_chain(chain, pos=-1, unique_name=True)
            new_structure.filename = f"{Path(self.filename).stem}_{chain.name}.pdb"
            structures.append(new_structure)
        return structures

    def to_file(self, filename: str = "") -> None:
        """Write the instance to a PDB or CIF file.

        Args:
            filename (str, optional): The name of the output file. If
            not provided, the instance's filename will be used.

        Raises:
            IOError: If the output format is not supported.

        Returns:
            None
        """
        if not filename:
            filename = self.filename
        else:
            self.filename = filename

        outtype = Path(filename).suffix[1:]
        if outtype == "pdb":
            self.structure.write_minimal_pdb(filename)
        elif outtype == "mmcif" or outtype == "cif":
            self.structure.make_mmcif_document().write_file(filename)
        else:
            raise IOError("Out format not supported, use pdb or mmcif")

    @requires_TEMPy
    def to_TEMPy(self, filename=None):
        """Converts the current structure to a TEMPy protein object.

        Args:
            filename (str, optional): The name of the output file. If not
            provided, the instance's filename will be used.

        Returns:
            A TEMPy protein object representing the current structure.

        Raises:
            None
        """
        if filename is None:
            filename = self.filename
        else:
            self.filename = filename
        data_block = self.__gemmi_structure.make_mmcif_document().sole_block()

        tempy_structure = mmCIFParser._convertGEMMItoTEMPy(
            data_block,
            self.__gemmi_structure,
            self.filename,
        )
        return tempy_structure
