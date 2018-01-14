# MultidirectionalAutoencoders
Image translation with multidirectional autoencoders that share a latent space.

Comprehensive documentation will follow...

# Dataset structure
The data and the corresponding label image need to have the same size.
<br/><br/>
/dataset/<br/>
----/trainA <- data_x.png data part of the image translation<br/>
----/trainB <- label_x.png label part to the corresponding data_x.png in trainA<br/>
----/testA <- data for testing, otherwise same as trainA<br/>
----/testB <- labels for testing, otherwise same as trainB<br/>
