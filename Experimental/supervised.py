import tensorflow as tf
import time
import cifar10


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './logs/training',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
def main(argv=None):
    cifar10.maybe_download_and_extract()
    global_step = tf.train.get_or_create_global_step()

    with tf.device('/cpu:0'):
        images, labels = cifar10.inputs(eval_data=False)
    
    logits = cifar10.inference(images)

    loss = cifar10.loss(logits, labels)

    train_op = cifar10.train(loss, global_step)

    class train_log_session_hook(tf.train.SessionRunHook):
        def after_run(self, run_context, run_values):
            if self.step % FLAGS.log_frequency == 0:
                time_delta = time.time() - self.start_t
                loss = run_values.results
                rate = FLAGS.log_frequency * FLAGS.batch_size / time_delta
                t_per_batch = float(time_delta/FLAGS.log_frequency)

                format_str = 'step %d, loss = %.2f w/ %.1f ex/sec and %.3f sec/batch'
                print(format_str % (self.step, loss, rate, t_per_batch))

                self.start_t = time.time()

        def before_run(self, run_context):
            self.step += 1
            return tf.train.SessionRunArgs(loss)

        def begin(self):
            self.step = 1
            self.start_t = time.time()


    with tf.train.MonitoredTrainingSession(
        checkpoint_dir = FLAGS.train_dir,
        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                 tf.train.NanTensorHook(loss),
                 train_log_session_hook()
                 ],
        config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        ) as mts:
        while not mts.should_stop():
            mts.run(train_op)


    
if __name__ == '__main__':
    tf.app.run()
