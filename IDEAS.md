The following are ideas categorized by type: should have, could have (it should be clarified if it is desireable from a technical or DX standpoint), and nice to have.

# Should have
- a mode where the dataloader is a thin wrapper over the webdatasets library (ie not using our custom shardcache and mixing source, rather use plain webdatasets, eg MultiShardSource or RandomMix), eventually leveraging torchdata.nodes for seamless switch. I can provide the full source of webdatasets if needed, just prompt me to add to your context.

# Could have
- leverage external libraries to increase performance or reduce loc of this library, maybe: ZMQ, webdatasets
- write performance critical code in a faster language, eg C++, Rust, etc. (this is a big one, I'm not sure if it's worth it)
- 

# Nice to have
- add support for other data sources, S3 in particular. Maybe to be done in dino_datasets ? Not useful at the moment anyway.
- expose augmentations for use in other libraries, eg torchvision (collate_fn and/or transform ?) to be discussed.