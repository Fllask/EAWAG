  *	ffffn??@2?
TIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::Filter::ParallelMapV2ꕲq?X@!?E?ͥ?X@)ꕲq?X@1?E?ͥ?X@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::Filter??h oAY@!??g?e?X@)sh??|???1h`9	???:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::Filter::ParallelMapV2::Shuffle??#?????!??m=N??)??#?????1??m=N??:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch? ?	???!?n??S1??)? ?	???1?n??S1??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchK?=?U??!?!?3y???)K?=?U??1?!?3y???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismh??|?5??!y7?H???)?ZӼ???1????Ì?:Preprocessing2F
Iterator::ModelL7?A`???!?ߩsڵ??)y?&1?l?156?0c[l?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.