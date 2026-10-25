#pragma once

#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>

/*template <typename T>
class SafeQueue
{
private:
    std::queue<T> process_queue;
    std::mutex queue_lock;
    std::condition_variable cv;

public:

    void push(const T& item)
    {
        std::lock_guard<std::mutex> lock(queue_lock);
        process_queue.push(item);
        cv.notify_one();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(queue_lock);

        cv.wait(lock, [this]{
            return !process_queue.empty();
        });

        T item = process_queue.front();
        process_queue.pop();

        return item;
    }
};*/

#pragma once
template <typename T>
class CircularBuffer
{
	private:
		std::vector<T> buffer;
		size_t capacity;
		alignas(64) std::atomic<size_t> writePtr{0};
		alignas(64) std::atomic<size_t> readPtr{0};
		
	public:
		//Constructor for initializing the buffer
		CircularBuffer(size_t size){
			capacity=size;
			buffer.resize(capacity);
		}
		
		bool push(const T& item)
		{    
			size_t w= writePtr.load();
			size_t r= readPtr.load();
			//check if the buffer is full (note, one slot may be sacrificed to check for full// work on this later on)
			if( (w+1)%capacity == r){
				return false;
			}
			
			buffer[w]=item;
			writePtr.store((w+1)%capacity, std::memory_order_release);
			return true;
		}
		
		bool pop(T& item)
		{	
			size_t w= writePtr.load(std::memory_order_acquire);
			size_t r= readPtr.load();
			//check if buffer is empty
			if(w==r){
				return false;
			}
			
			item= buffer[r];
			readPtr.store((r+1)%capacity);
			
			return true;
		}
		
		bool is_empty(){
			size_t w= writePtr.load();
			size_t r= readPtr.load();
			return (w==r);
		}
		
		bool is_full(){
			size_t w= writePtr.load();
			size_t r= readPtr.load();
			return ((w+1)%capacity == r);
		}
		
};




		
//
