project(
    'NonLinearLeastSquare',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)
eigen_dep = dependency('eigen3')
subdir('lib')

executable('NonLinearLeastSquare',
    [
        'src/NonLinearLeastSquare.cpp',
        xxd.process('src/NonLinearLeastSquare.cl'),
    ],
    extra_files: [
        'src/NonLinearLeastSquare.cl',
    ],
    dependencies : [
        core_dep,
        opencl_dep,
        eigen_dep,
    ],
)
