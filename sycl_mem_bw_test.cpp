#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl; // (optional) avoids need for "sycl::" before SYCL names

double timeit(event& e){
    auto start_time = e.get_profiling_info<info::event_profiling::command_start>();
    auto end_time = e.get_profiling_info<info::event_profiling::command_end>();
    auto exec_time = (end_time - start_time)/1000000000.0;//second
    return exec_time;
}

template<typename T>
void copy_v1(T* in, T*out, unsigned long N, queue & q){
   auto e = q.submit([&](auto &h){
       sycl::stream cout(65536, 256, h);
       h.parallel_for(sycl::nd_range(sycl::range{N/16}, sycl::range(32)),
                    [=](sycl::nd_item<1>it){
                         auto i = it.get_global_linear_id();
                         auto sg = it.get_sub_group();
                         auto sgSize = sg.get_local_range()[0];
                         auto laneid=sg.get_local_id()[0];
                         constexpr int  vec_size =4;
                         i = (i/sgSize)*sgSize*4 + laneid;
                         //cout<<"i:"<<i<<sycl::endl;
                         auto out_vec = reinterpret_cast<sycl::vec<T, vec_size>*>(out);
                         auto in_vec = reinterpret_cast<sycl::vec<T, vec_size>*>(in);
                         for(int j=0; j<=sgSize*16; j+=sgSize){
                            out_vec[i+j]=in_vec[i+j];

                            //if(i==0) cout<<"j"<<j<<sycl::endl;
                         }
                         /*i = (i/sgSize)*sgSize*16 + laneid*vec_size;
                         for (int j = 0; j < 4; j++) {
                             sycl::vec<int, 4> x;
                             sycl::vec<int, 4> *q =
                                 (sycl::vec<int, 4> *)(&(in[i + j * sgSize * 4]));
                             x = *q;
                             sycl::vec<int, 4> *r =
                                 (sycl::vec<int, 4> *)(&(out[i + j * sgSize * 4]));
                             *r = x;
                         }*/
                     });
   });
   q.wait();
   auto exec_time = timeit(e);
   std::cout <<"Vec R/w BW:" <<2*N*sizeof(T)/exec_time/1000000000 <<"GB/s"<<std::endl;
}

template<typename T>
void copy_v2(const T* in, T*out, unsigned long N, queue & q){
   auto e = q.submit([&](auto &h){
       h.parallel_for(sycl::nd_range(sycl::range{N/16}, sycl::range(32)),
                      [=](sycl::nd_item<1>it){
                         auto i = it.get_global_linear_id();
                         auto gid=i;
                         auto sg = it.get_sub_group();
                         auto sgSize = sg.get_local_range()[0];
                         auto laneid=sg.get_local_id()[0];
                         i = (i/sgSize)*sgSize*16 + laneid;
                         for(int j=0; j<=sgSize*16; j+=sgSize){
                            out[i+j]=in[i+j]+gid;
                         }
                     });
   });
   q.wait();
   auto exec_time = timeit(e);
   std::cout <<"w/O Vec R/W BW:" <<2*N*sizeof(T)/exec_time/1000000000 <<"GB/s"<<std::endl;
}

template<typename T>
void copy_cpu(const T* in, T*out, unsigned long N){
    for(auto i=0; i<N; i++){
       out[i]=in[i];
    }
}

template<typename T>
bool acc_test(const T* in, const T*out, unsigned long N){
   for(auto i=0; i<N; i++){
      if(in[i]-out[i]>0.0000001){printf("acc fail"); return -1;}
   }
   return 0;
}

int main() {
  constexpr int N = 256 * 1024 * 1024;
  using scalar_t = float;
  queue q(sycl::property_list{sycl::property::queue::enable_profiling()});
  scalar_t*in =  sycl::malloc_shared<scalar_t>(N*sizeof(scalar_t), q);//sycl::aligned_alloc_device<scalar_t>(512, N*sizeof(scalar_t), q);
  scalar_t*out_v2 = sycl::malloc_shared<scalar_t>(N*sizeof(scalar_t), q);//sycl::aligned_alloc_device<scalar_t>(512, N*sizeof(scalar_t), q);
  scalar_t*out_v1 = sycl::malloc_shared<scalar_t>(N*sizeof(scalar_t), q);//sycl::aligned_alloc_device<scalar_t>(512, N*sizeof(scalar_t), q);
  memset(in, 0, N*sizeof(scalar_t));
  scalar_t *out_cpu=static_cast<scalar_t*>(malloc(N*sizeof(scalar_t)));
  //scalar_t *out_cpu_1=static_cast<scalar_t*>(malloc(N*sizeof(scalar_t)));
  //scalar_t *in_cpu=static_cast<scalar_t*>(malloc(N*sizeof(scalar_t)));
  copy_v2(in, out_v2, N, q);
  //copy_v2(out, in, N, q);
  copy_cpu(out_v2, out_cpu, N);
  copy_v1(out_v2, out_v1, N, q);
  //q.memcpy(out_cpu_1, out, N*sizeof(scalar_t)).wait();
  //out_cpu[0]=9;
  /*std::cout<<"#################out_v1:\n";
  for(int i=0; i<N; i++){
     std::cout << out_v1[i]<<" ";
  }
  std::cout<<"\n#################out_v2:\n";
  for(int i=0; i<N; i++){
     std::cout << out_v2[i]<<" ";
  }*/

  acc_test(out_v2, out_v1, N);
  for(int i=0; i<10; i++){
      copy_v1(in, out_v1, N, q);
  }
  for(int i=0;i<10; i++){
     copy_v2(in, out_v2, N, q);
  }
  /*for(int i=0; i<N; i++){
     std::cout << out_cpu[i]<<" ";
  }*/
  return 0;
}
