**TODO LIST**

1. **Add intensity, linearity, sphericity and planarity 
as input features. 
(features = {height, intensity, linearity, sphericity, planarity})
 Use `.h5` file to store.**

2. Apply Superpoint & Superglue to birdview point cloud 
birdview operation
    
    * Feature detection (to look up)    
    
    * **Feature detection: use Superpoint, with height, intensity,
    occupancy binary value**  
    
    * Feature description and matching (almost done)
 
3. Implement a tiny network to judge whether a birdview
image is salient for global localization / loop closure
detection

4. Lazy triplet loss, lazy quadruplet loss

