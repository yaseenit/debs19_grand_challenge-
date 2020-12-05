/*
Passing variables / arrays between cython and cpp
Example from 
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

Adapted to include passing of multidimensional arrays

*/

#include <vector>
#include <iostream>

namespace pclwrapper {
    class  Wrapper{
    public:
        Wrapper();
        ~Wrapper();
         std::vector<long double>  compute(std::vector< std::vector< double> > sv);
    };
}