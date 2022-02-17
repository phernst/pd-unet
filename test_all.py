from test_pd_orig import main as pd_orig
from test_pd_unet_sino import main as pd_unet_sino
from test_pd_unet import main as pd_unet
from test_reco_unet import main as reco_unet
from test_sino_unet import main as sino_unet


def test_all(parfan: str, subtype: str, subnum: int):
    print("pd_orig")
    pd_orig(parfan, subtype, subnum)
    print("pd_unet_sino")
    pd_unet_sino(parfan, subtype, subnum)
    print("pd_unet")
    pd_unet(parfan, subtype, subnum)
    print("reco_unet")
    reco_unet(parfan, subtype, subnum)
    print("sino_unet")
    sino_unet(parfan, subtype, subnum)


if __name__ == '__main__':
    test_all('parallel', 'sparse', 4)
    test_all('parallel', 'sparse', 8)
    test_all('parallel', 'sparse', 16)
    test_all('fan', 'sparse', 4)
    test_all('fan', 'sparse', 8)
    test_all('fan', 'sparse', 16)
