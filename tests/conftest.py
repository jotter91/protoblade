import pathlib
import pytest

fixture_dir = pathlib.Path(__file__).parent / "fixtures"

@pytest.fixture()
def naca0012_files():
    naca0012_dir = fixture_dir / "naca0012"
    return {
        'upper': naca0012_dir / "naca0012_upper.fpd",
        'lower': naca0012_dir / "naca0012_lower.fpd",
    }

@pytest.fixture()
def vki_files():
    vki_dir = fixture_dir / "vki_turbine"
    return {
        'pressure': vki_dir / "vki_ps.fpd",
        'suction' : vki_dir / "vki_ss.fpd"
    }

