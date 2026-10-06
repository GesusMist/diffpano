"""Explicit, supplementary seam metric conventions; see metric_protocols.json."""
import numpy as np
import cv2
from skimage.metrics import structural_similarity
from .render import edge_pairs,oriented_edge

DS_PROTOCOL=dict(name='DS',status='computed_adaptation',implementation='author-linked survey reimplementation; output singleton dimension corrected at call site',
    source='https://github.com/littlewhitesea/Text-Driven-Pano-Gen/blob/main/evaluation_metrics/Discontinuity_Score_cal.py',
    paper='https://arxiv.org/html/2407.18207',kernel=[[-3,0,3],[-10,0,10],[-3,0,3]],
    intensity='float32 RGB/255, grayscale 0.299R+0.587G+0.114B',resolution='native ERP 4096x2048; no resize',
    seam='last3 + first3 columns',vertical_padding='zero, matching linked implementation',c=.1,
    formula='mean_y((abs(g[:,1])/(abs(g[:,0])+c)+abs(g[:,2])/(abs(g[:,3])+c))/2); L/H=1',
    direction='lower',units='dimensionless ratio',constant_image=0.,
    limitation='Original author code unreleased. Linked reimplementation uses first-order Scharr; paper describes second-order Scharr. This is explicitly that linked variant, not a claim of exact original DS.')
SEAM_PROTOCOL=dict(status='computed_adaptation',paper='https://arxiv.org/html/2512.06885v1',equations=[11,12],
    representation='ERP-derived cubemap, six 512x512 faces, 90-degree FoV, roll0, pixel centers strictly inside each face',
    adjacency='12 unique edges derived from equal world-space edge endpoints; corresponding edge sample order reversed when needed',
    band_width=5,band_rule='round(0.01*512)',intensity='RGB uint8 [0,255]',
    ssim=dict(window_size=5,gaussian_weights=False,use_sample_covariance=True,K1=.01,K2=.03,data_range=255,channels='average RGB'),
    sobel=dict(operator='3x3 x-direction Sobel after orienting seam vertically',grayscale='OpenCV RGB2GRAY float32',
        border='BORDER_REPLICATE',scale=1,selected_column=0,formula='average over 12 edges of (mean(abs(left gradient))+mean(abs(right gradient)))/2'),
    limitation='Official code not released. Underspecified paper settings are fixed here and explicitly adaptations. Faces share one ERP source and are not independent generated views. Near-edge pixel centers differ; identical edge rays are not scored. Sobel measures edge-adjacent texture gradients, not a difference between gradients, and misses constant-valued face offsets.')

def ds(image):
    a=image.astype(np.float32)/255.
    g=.299*a[...,0]+.587*a[...,1]+.114*a[...,2]
    a=np.concatenate([g[:,-3:],g[:,:3]],axis=1)
    k=np.array(DS_PROTOCOL['kernel'],dtype=np.float32)
    # Equivalent to the linked torch conv2d padding=(1,0).
    response=cv2.filter2D(a,cv2.CV_32F,k,borderType=cv2.BORDER_CONSTANT)[:,1:-1]
    v=np.abs(response)
    return float(np.mean(v[:,1]/(v[:,0]+.1)+v[:,2]/(v[:,3]+.1))/2)

def cubemap_seams(faces):
    assert faces.shape==(6,512,512,3)
    scores=[];gradients=[]
    for a,b,reverse in edge_pairs():
        left=oriented_edge(faces[a[0]],a[1]);right=oriented_edge(faces[b[0]],b[1])
        if reverse:right=right[::-1]
        score=structural_similarity(left[:,:5],right[:,:5],win_size=5,channel_axis=2,data_range=255,
            gaussian_weights=False,use_sample_covariance=True,K1=.01,K2=.03)
        grad=[]
        for face in (left,right):
            grey=cv2.cvtColor(face.astype(np.float32),cv2.COLOR_RGB2GRAY)
            gx=cv2.Sobel(grey,cv2.CV_32F,1,0,ksize=3,scale=1,borderType=cv2.BORDER_REPLICATE)
            grad.append(float(np.abs(gx[:,0]).mean()))
        scores.append(float(score));gradients.append(sum(grad)/2)
    return {'Seam-SSIM':float(np.mean(scores)),'Seam-Sobel':float(np.mean(gradients)),
            'edge_ssim':scores,'edge_sobel':gradients}
