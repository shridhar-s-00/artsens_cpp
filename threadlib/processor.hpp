#pragma once
#include <thread>
#include "threadqueue.hpp"
#include <iostream>
#include <atomic>
#include <stdexcept>

class Processor
{
	protected:
		std::thread worker_thread;
		std::atomic<bool> running{false};
		
	public:

		void start(){
			
			//size_t run_flag= running.load();
			
			if(running.load())
				throw std::runtime_error("Thread is already running");
			
			running.store(true);
			
			worker_thread= std::thread(&Processor::run,this);
			#pragma once
		}
		
		void stop(){
			if(worker_thread.joinable()){
				running.store(false);
				worker_thread.join();
			}
		}
		
		virtual ~Processor(){
			stop();
		}
		
		virtual void run() =0;
		
};

template <typename T>
class SourceProcessor:public Processor
{
	private:
		CircularBuffer<T> *outputbuffer;
	public:
		
		SourceProcessor(CircularBuffer<T> *out){
		
			outputbuffer=out;
		}
		
		void run() override{
			while(running.load()){
				T item= process();
				/*if(outputbuffer && (!outputbuffer->is_full()))
					outputbuffer->push(item);*/
					
					while(running && !outputbuffer->push(item)){
					
					    std::this_thread::yield(); //making the thread give up cpu so the other process can drain the queue
					}
			}
			
		}
		
		virtual T process()=0;
		
};

	








template <typename Tin, typename Tout>
class ConnectProcessor : public Processor
{
private:
    CircularBuffer<Tin>* inputbuffer;
    CircularBuffer<Tout>* outputbuffer;

public:
    ConnectProcessor(CircularBuffer<Tin>* in, CircularBuffer<Tout>* out)
        : inputbuffer(in), outputbuffer(out) {}

    void run() override
    {
        Tin item;

        while (running.load())
        {
            if (inputbuffer && inputbuffer->pop(item))
            {
                Tout result = process(item);

                while (running && outputbuffer && !outputbuffer->push(result))
                {
                    std::this_thread::yield();
                }
            }
        }
    }

    virtual Tout process(const Tin& item) = 0;
};








template <typename T>
class SinkProcessor:public Processor
{
	private:
		CircularBuffer<T> *inputbuffer;
		
	public:
		
		SinkProcessor(CircularBuffer<T> *in){
			inputbuffer=in;
			
		}
		
		/*void run() override{
			T item ;
			while(running.load()){
				if(inputbuffer && (!inputbuffer->is_empty()))
					inputbuffer->pop(item);
				
				 process(item);
			
			}
			
		}*/
		
		
		void run() override
		{
		    T item;

		    while(running.load())
		    {
			if(inputbuffer && inputbuffer->pop(item))
			{
			    process(item);
			}
		    }
		}
		
		virtual void process(const T& item)=0;
		
		
};

		
		
		
