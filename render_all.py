from test_pd_orig import render_all_images as pd_orig
from test_pd_unet_sino import render_all_images as pd_unet_sino
from test_pd_unet import render_all_images as pd_unet
from test_reco_unet import render_all_images as reco_unet
from test_sino_unet import render_all_images as sino_unet


def render_all():
    print("pd_orig")
    pd_orig()
    print("pd_unet_sino")
    pd_unet_sino()
    print("pd_unet")
    pd_unet()
    print("reco_unet")
    reco_unet()
    print("sino_unet")
    sino_unet()


if __name__ == '__main__':
    render_all()
