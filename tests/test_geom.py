import scipy.interpolate

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

def test_split_into_le_main_te(vki_files):
    ps_pnts = geom.load_curves_from_fpd(vki_files['pressure'])
    #ps_pnts['y'] = -ps_pnts['y']
    ss_pnts = geom.load_curves_from_fpd(vki_files['suction'])
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(ss_pnts['x'],ss_pnts['y'])
    plt.savefig('fisrt_blade.png')

    pnts = ss_pnts

    #calculate curvature
    s = geom.calculate_curve_length(pnts['x'],pnts['y'])

    dy_dx = np.diff(s)/np.diff(pnts['x'])
    d2y_dx = np.diff(dy_dx)/np.diff(pnts['x'][:-1])
    grad_s = np.gradient(s)
    sec_s = np.gradient(grad_s)

    curv = sec_s/((1.0+grad_s**2)**1.5)


    plt.figure()
    plt.plot(pnts['x'][:-2])#,d2y_dx)
    plt.savefig('dydx.png')

    #s_fit = scipy.interpolate.CubicSpline(s,pnts['x'])

    plt.figure()
    plt.plot(s,pnts['x'])
    plt.savefig('curv.png')

    #s_grad = s_fit.derivative(2)

    #plt.figure()
    #plt.plot(s,s_grad(pnts['x']))
    #plt.savefig('s_grad.png')



    plt.figure()
    plt.plot(pnts['x'], grad_s )
    plt.savefig('grad.png')


    plt.figure()
    plt.plot(pnts['x'],curv )
    plt.ylim([-1e-7,1e-7])
    plt.savefig('curvature.png')

    #look for sign change
    idx1 = np.where(curv[:-1] * curv[1:] < 0)[0] + 1

    plt.figure()
    plt.plot(pnts['x'],pnts['y'],'-')
    plt.plot(pnts['x'][idx1][0],pnts['y'][idx1][0],'x')


    #look for sign change in x
    diff_x = np.diff(pnts['x'])
    idx2 = np.where(np.diff(np.sign(diff_x)) != 0)[0] + 1
    plt.plot(pnts['x'][idx2][-1],pnts['y'][idx2][-1],'x')
    plt.savefig('blade.png')

    plt.figure()
    plt.plot(diff_x)
    plt.savefig('diff_x.png')