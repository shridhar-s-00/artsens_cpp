#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <vector>
#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/data_structures/ring_buffer.hpp"











void test_float_fifo()
{
    Ringbuffer_handle rb;

    RingbufferCreate(
        &rb,
        5,
        sizeof(float)
    );

    float a=10.5f;
    float b=20.5f;
    float c=30.5f;

    assert(RingbufferEnqueue(&rb,(uint8_t*)&a));
    assert(RingbufferEnqueue(&rb,(uint8_t*)&b));
    assert(RingbufferEnqueue(&rb,(uint8_t*)&c));

    float out;

    assert(RingbufferDequeue(&rb,(uint8_t*)&out));
    assert(out==10.5f);

    assert(RingbufferDequeue(&rb,(uint8_t*)&out));
    assert(out==20.5f);

    assert(RingbufferDequeue(&rb,(uint8_t*)&out));
    assert(out==30.5f);

    printf("PASS test_float_fifo\n");
}



void test_wraparound()
{
    Ringbuffer_handle rb;

    RingbufferCreate(
      &rb,
      3,
      sizeof(int)
    );

    int a=1,b=2,c=3;
    int out;

    assert(RingbufferEnqueue(&rb,(uint8_t*)&a));
    assert(RingbufferEnqueue(&rb,(uint8_t*)&b));

    assert(RingbufferDequeue(&rb,(uint8_t*)&out));
    assert(out==1);

    assert(RingbufferDequeue(&rb,(uint8_t*)&out));
    assert(out==2);

    // should wrap head
    assert(RingbufferEnqueue(&rb,(uint8_t*)&c));

    assert(RingbufferDequeue(&rb,(uint8_t*)&out));
    assert(out==3);

    printf("PASS test_wraparound\n");
}



void test_full_queue()
{
    Ringbuffer_handle rb;

    RingbufferCreate(
        &rb,
        2,
        sizeof(int)
    );

    int a=10,b=20,c=30;

    assert(
      RingbufferEnqueue(
        &rb,
        (uint8_t*)&a
      )
    );

    assert(
      RingbufferEnqueue(
        &rb,
        (uint8_t*)&b
      )
    );

    // should fail
    assert(
      RingbufferEnqueue(
        &rb,
        (uint8_t*)&c
      ) == false
    );

    printf("PASS test_full_queue\n");
}



void test_empty_queue()
{
    Ringbuffer_handle rb;

    RingbufferCreate(
      &rb,
      3,
      sizeof(int)
    );

    int out;

    assert(
      RingbufferDequeue(
         &rb,
         (uint8_t*)&out
      ) == false
    );

    printf("PASS test_empty_queue\n");
}



struct Vec3
{
   float x;
   float y;
   float z;
};

void test_struct_enqueue()
{
    Ringbuffer_handle rb;

    RingbufferCreate(
      &rb,
      4,
      sizeof(Vec3)
    );

    Vec3 v1={1,2,3};
    Vec3 v2={4,5,6};

    assert(
      RingbufferEnqueue(
         &rb,
         (uint8_t*)&v1
      )
    );

    assert(
      RingbufferEnqueue(
         &rb,
         (uint8_t*)&v2
      )
    );

    Vec3 out;

    assert(
      RingbufferDequeue(
         &rb,
         (uint8_t*)&out
      )
    );

    assert(out.x==1);
    assert(out.y==2);
    assert(out.z==3);

    assert(
      RingbufferDequeue(
         &rb,
         (uint8_t*)&out
      )
    );

    assert(out.x==4);
    assert(out.y==5);
    assert(out.z==6);

    printf("PASS test_struct_enqueue\n");
}



void test_stress()
{
    Ringbuffer_handle rb;

    RingbufferCreate(
      &rb,
      8,
      sizeof(int)
    );

    for(int i=0;i<10000;i++)
    {
        assert(
          RingbufferEnqueue(
             &rb,
             (uint8_t*)&i
          )
        );

        int out;

        assert(
          RingbufferDequeue(
             &rb,
             (uint8_t*)&out
          )
        );

        assert(out==i);
    }

    printf("PASS test_stress\n");
}

#include <array>


struct BigVector
{
    std::array<float,1024> data;
};



void test_big_vector_payload()
{
    Ringbuffer_handle rb;

    RingbufferCreate(
      &rb,
      3,
      sizeof(BigVector)
    );

    BigVector in1;
    BigVector in2;

    for(int i=0;i<1024;i++)
    {
        in1.data[i]=i*1.0f;
        in2.data[i]=1000+i;
    }


    assert(
      RingbufferEnqueue(
         &rb,
         (uint8_t*)&in1
      )
    );

    assert(
      RingbufferEnqueue(
         &rb,
         (uint8_t*)&in2
      )
    );


    BigVector out;


    assert(
      RingbufferDequeue(
         &rb,
         (uint8_t*)&out
      )
    );


    for(int i=0;i<1024;i++)
    {
        assert(
           out.data[i]==in1.data[i]
        );
    }



    assert(
      RingbufferDequeue(
         &rb,
         (uint8_t*)&out
      )
    );


    for(int i=0;i<1024;i++)
    {
        assert(
          out.data[i]==in2.data[i]
        );
    }

    printf(
      "PASS test_big_vector_payload\n"
    );
}

int main()
{
    test_float_fifo();

    test_wraparound();

    test_full_queue();

    test_empty_queue();

    test_struct_enqueue();

    test_stress();
    
    test_big_vector_payload();

    printf("\nALL TESTS PASSED\n");
}
	
