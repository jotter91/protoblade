from protoblade import geom
import numpy as np

def test_read_from_fpd(naca0012_files):
    for file in naca0012_files.values():
        pnts = geom.load_curves_from_fpd(file)

        for item in ['x','y','z']:
            assert len(pnts[item])==66

def test_reinterpolate_curve(naca0012_files):
    naca = geom.load_curves_from_fpd(naca0012_files['upper'])
    s = geom.calculate_curve_length(naca['x'],naca['y'])
    x,y,s_new = geom.reinterpolate_curve(naca['x'],naca['y'],s)

    plot = False
    if plot:
        from matplotlib import pyplot
        pyplot.figure()
        pyplot.plot(naca['x'],naca['y'],'x')
        pyplot.plot(x,y,'o')
        pyplot.show()
        pyplot.savefig("mygraph.png")
    return

def test_convert_to_polar():
    pts = np.array([(1.,1.,1.),(2.,2.,2.),(3.,3.,3.)],dtype=geom.cartesian_type)

    polar = geom.convert_to_polar(pts)

    np.testing.assert_array_equal(polar['r'],np.array([2.0**0.5,8.0**0.5,18**0.5]))
    np.testing.assert_array_equal(polar['theta'],np.array([np.pi/4]*3))
    np.testing.assert_array_equal(polar['z'],np.array([1.0,2.0,3.0]))

def test_calculate_curve_length():
    N=20
    x = np.linspace(0,1,N)
    y = np.linspace(1,2,N)

    s = geom.calculate_curve_length(x,y)

    np.testing.assert_array_almost_equal(s,np.asarray([0.,.07443229,0.14886459,0.22329688,0.29772917,0.37216146,
                                                0.44659376, 0.52102605, 0.59545834, 0.66989063, 0.74432293,
                                                0.81875522, 0.89318751, 0.96761981, 1.0420521,  1.11648439,
                                                1.19091668, 1.26534898, 1.33978127, 1.41421356]))