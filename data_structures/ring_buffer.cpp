#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <vector>


struct Ringbuffer_handle{

	uint8_t* buffer; 
	size_t size;
	size_t queue_length;
	size_t count;
	int head;
	int tail;
};



void RingbufferCreate(Ringbuffer_handle* handle,size_t queue_length, size_t item_size){

           
	handle->buffer= (uint8_t*)malloc(item_size*queue_length);
	
	handle->size= item_size;
	handle->queue_length= queue_length;
	handle->count= 0;
	handle->head=0;
	handle->tail=0;
}


bool RingbufferEnqueue(Ringbuffer_handle* handle,uint8_t* data){// assuming spsc we do not use locks

	if(handle->count==handle->queue_length){
		printf("\n queue full!\n");
		return false;
	}
	
	size_t queue_size= handle->size * handle->queue_length;
	
	memcpy((handle->buffer + handle->head),data,handle->size);
	handle->head= (handle->head + handle->size)%queue_size;
	(handle->count)++;
	
	return true;
	
}

bool RingbufferDequeue(Ringbuffer_handle* handle,uint8_t* data){

	if(handle->count==0){
	
		return false;
	}
	
	size_t queue_size= handle->size * handle->queue_length;
	
	memcpy(data,(handle->buffer + handle->tail),handle->size);
	handle->tail= (handle->tail + handle->size)%queue_size;
	(handle->count)--;
	
	return true;
	
}


