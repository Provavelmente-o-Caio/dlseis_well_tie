from interface import PetrelInterface
import sys

testes_interface = PetrelInterface(sys.argv[1])

testes_interface.auto_well_tie()
