Faster eigen values computation: 
    
* use Eigen::SelfAdjointEigenSolver

* the mkpts coordinates of SuperGlue are (u,v) associated to (y,x):

    * (x,y) = (u,v) * meters_per_pixel - 50
    
    * (u,v) = ((x,y) + 50) / meters_per_pixel

* import cv2 in global_localization.py will cause error?



