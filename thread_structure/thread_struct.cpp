#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>
#include <pthread.h>
#include <unistd.h>


#include "/home/act/Vessel_Vision_live_track/project_version_v1r0/data_structures/ring_buffer.hpp"


Ringbuffer_handle handle_rb,handle_rb1;

void *generate_numbers(void *args)
{
    long long counter = 0;

    while (1)
    {
        if (RingbufferEnqueue(&handle_rb,(uint8_t*)&counter))
        {
            counter++;
            
        }
        //usleep(100);
    }

    return NULL;
}

void *square_numbers(void *args)
{
    long long rx;

    while (1)
    {
        if (RingbufferDequeue(&handle_rb,(uint8_t*)&rx))
        {
            rx *= rx;

            while(!RingbufferEnqueue(&handle_rb1,(uint8_t*)&rx))
                ;
        }
    }

    return NULL;
}

void *print_numbers(void *args)
{
    long long rx;

    while (1)
    {
        if (RingbufferDequeue(&handle_rb1,(uint8_t*)&rx))
        {
            printf("%lld\n", rx);
        }
    }

    return NULL;
}

/*
int main(){


	pthread_t tid_gen, tid_sq, tid_print;
	
	
	
	RingbufferCreate(&handle_rb,10,sizeof(long long int));
	RingbufferCreate(&handle_rb1,10,sizeof(long long int));
	
	
	
	int ret1= pthread_create(&tid_gen,NULL,generate_numbers,NULL);
	int ret2= pthread_create(&tid_sq,NULL,square_numbers,NULL);
	int ret3= pthread_create(&tid_print,NULL,print_numbers,NULL);
	
	ret1= pthread_join(tid_gen,NULL);
	ret2= pthread_join(tid_sq,NULL);
	ret3= pthread_join(tid_print,NULL);
	
	return 0;	
}*/
	
	
	
	
	
	
	





