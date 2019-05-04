import tensorflow as tf
sess=tf.Session()
def run(a):
    sess.run(tf.global_variables_initializer())
    return sess.run(a)
def get_scope_variable(scope_name,var_name,shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            var=tf.get_variable(var_name,shape)
        except ValueError:
            print('reuse variables here')
            scope.reuse_variables()
            var=tf.get_variable(var_name)
    return var
var_1 = get_scope_variable("cur_scope","my_var",[100])
var_2 = get_scope_variable("cur_scope","my_var",[100])
print(var_1 is var_2)
print(var_1.name)