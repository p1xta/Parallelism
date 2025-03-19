#include <iostream>
#include <chrono>
#include <thread>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <cmath>
#include <future>
#include <functional>
#include <condition_variable>
#include <fstream>
#include <random>

template<typename T>
class safeThreadQueue {
public:
    void push(T item) {
        std::unique_lock<std::mutex> lock(mtx);
        st_queue.push(std::move(item));
        cv.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !st_queue.empty(); });
        T item = std::move(st_queue.front());
        st_queue.pop();
        return item;
    }
    
    bool empty() {
        std::unique_lock<std::mutex> lock(mtx);
        return st_queue.empty();
    }
private:
    std::queue<T> st_queue;
    std::mutex mtx;
    std::condition_variable cv;
};

template<typename T>
class Server {
public:
    Server() : running(false), task_id(0) {}
    ~Server() {}

    void start() {
        running = true;
        server_thread = std::thread(&Server::perform_tasks, this);
        std::cout << "Server started" << std::endl;
    }

    void stop() {
        running = false;
        cv.notify_all();
        if (server_thread.joinable()) {
            server_thread.join();
        }
        std::cout << "Server stopped" << std::endl;
    }

    int add_task(std::packaged_task<T()>&& task) {
        std::lock_guard<std::mutex> lock(mtx);
        int current_id = this->task_id++;
        std::future<T> task_result = task.get_future();
        tasks_queue.push(std::move(task));
        results[current_id] = std::move(task_result);
        cv.notify_one(); 
        return current_id;
    }

    T request_result(int id_res) {
        auto res_it = results.find(id_res);
        if (res_it != results.end()){
            res_it->second.wait();
            return res_it->second.get();
        }
        throw std::runtime_error("No such task id");
    }

private:
    void perform_tasks() {
        while(true) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return !running || !tasks_queue.empty(); });
            if (!running && tasks_queue.empty()) {
                break;
            }
            if (!tasks_queue.empty()) {
                std::packaged_task<T()> task = tasks_queue.pop();
                lock.unlock();
                task();
            }
        }
    }
    bool running;
    std::thread server_thread;
    safeThreadQueue<std::packaged_task<T()>> tasks_queue;
    std::unordered_map<int, std::future<T>> results;
    std::mutex mtx;
    std::condition_variable cv;
    int task_id;
};

template <typename T>
T Tsin(T x) {
    return std::sin(x);
}

template <typename T>
T Tsqrt(T x) {
    return std::sqrt(x);
}

template <typename T>
T Tpow(T x, T y) {
    return std::pow(x, y);
}

template <typename T>
class Client {
public:
    Client () = default;
    Client (Server<T>& server, int task_num, const std::string file_name, const std::string task_type) 
        : server(server), task_number(task_num), output_filename(file_name), task_type(task_type) {}
    ~Client() {}

    void create_tasks() {
        std::ofstream output_file(output_filename);
        output_file << task_type << "\n\n";
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<T> dist(1.0,50.0);
        for (int i = 0; i < task_number; i++) {
            std::packaged_task<T()> task;
            T x = dist(rng);  T y;
            if (task_type == "sin") {
                task = std::packaged_task<T()>(std::bind(Tsin<T>, x));
            }
            else if (task_type == "sqrt") {
                task = std::packaged_task<T()>(std::bind(Tsqrt<T>, x));
            }
            else if (task_type == "pow") {
                y = dist(rng); 
                task = std::packaged_task<T()>(std::bind(Tpow<T>, x, y));
            }
            else
                throw std::invalid_argument("No such task type");

            int task_id = server.add_task(std::move(task));
            T result = server.request_result(task_id);
            if (task_type == "pow") 
                output_file << "Task id " << task_id << " = (" << x << ", " << y << ") =  " << result << "\n"; 
            else
                output_file << "Task id " << task_id << " = (" << x << ") =  " << result << "\n"; 
        }
        output_file.close();
    }

private:
    Server<T>& server;
    int task_number;
    std::string task_type;
    std::string output_filename;
};

int main() {

    Server<double> server;
    server.start();

    Client<double> client1(server, 10, "client1.txt", "sin");
    Client<double> client2(server, 15, "client2.txt", "sqrt");
    Client<double> client3(server, 20, "client3.txt", "pow");
    
    std::vector<std::thread> threads;
    threads.emplace_back(&Client<double>::create_tasks, &client1);
    threads.emplace_back(&Client<double>::create_tasks, &client2);
    threads.emplace_back(&Client<double>::create_tasks, &client3);

    for (auto& thread : threads)  
        thread.join();
    
    server.stop();
    return 0;
}