#include "crow_all.h"
#include "httplib.h"
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

struct Truck {
    std::string id;
    int capacity;
    double cost_per_trip;
    bool available;
};

std::vector<Truck> g_park = {
        {"small_truck", 150, 300.0, true},
        {"small_truck", 150, 300.0, true},
        {"medium_track", 400, 650.0, false},
        {"medium_track", 400, 650.0, true},
        {"eco_medium_track", 350, 500.0, true},
        {"big_truck", 600, 1000.0, true}
    };

const std::string ml_host = "localhost";
const int ml_port = 5000;

std::string call_ml(const std::string& body) {
    httplib::Client client(ml_host, ml_port);
    client.set_read_timeout(5);
    client.set_write_timeout(5);

    auto res = client.Post("/same_as_yours", body, "application/json");

    if (res && res->status == 200){
        return res->body;
    } else {
        return "{\"predicted_volume\": 0}";
    }
}

struct Selected_trucks {
    std::vector<std::string> selected_truck_ids;
    int tot_capacity;
    double tot_cost;
    double usage_percent;

    Selected_trucks():
        tot_capacity(0), tot_cost(0), usage_percent(0) {}
};

Selected_trucks select_trucks(double predicted_volume, const std::vector<Truck>& park, 
    bool optimize_by_cost){
    Selected_trucks ans;

    std::vector<Truck> available_trucks;
    for(const auto& truck : park){
        if (truck.available){
            available_trucks.push_back(truck);
        }
    }

    if (optimize_by_cost){
        std::sort(available_trucks.begin(), available_trucks.end(),
    [](const Truck& a, const Truck& b){
        double coef_a = a.cost_per_trip/a.capacity;
        double coef_b = b.cost_per_trip/b.capacity;
        return coef_a < coef_b;
    });
    } else {
        std::sort(available_trucks.begin(), available_trucks.end(),
    [](const Truck& a, const Truck& b){
        return a.capacity > b.capacity;
    });
    }

    double remaining = predicted_volume;

    for (const auto& truck : available_trucks) {
        if (remaining <= 0) break;
        
        ans.selected_truck_ids.push_back(truck.id);
        ans.tot_capacity += truck.capacity;
        ans.tot_cost += truck.cost_per_trip;
        remaining -= truck.capacity;
    }

    if (ans.tot_capacity > 0){
        ans.usage_percent = std::round((predicted_volume / ans.tot_capacity) * 100);
    }

    return ans;
}

int main(){
    crow::SimpleApp app;

    const bool optimize_by_cost = false;

    CROW_ROUTE(app, "/api/calculate").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req){
        crow::json::wvalue response;

        try {
            auto input_data = crow::json::load(req.body);

            if (!input_data) {
                response["status"] = "error";
                response["message"] = "invalid json input";
                return crow::response(400, response.dump());
            }

            crow::json::wvalue ml_request;
            ml_request["route_id"] = input_data["route_id"];
            ml_request["office_from_id"] = input_data["office_from_id"];
            ml_request["timestamp"] = input_data["timestamp"];

            crow::json::wvalue statuses;
            for(int i = 1; i <= 8; i++){
                std::string key = "status_" + std::to_string(i);
                    statuses[key] = input_data[key];
            }

            ml_request["statuses"] = std::move(statuses);

            std::string ml_request_str = ml_request.dump();

            std::string ml_response_str = call_ml(ml_request_str);

            auto ml_response = crow::json::load(ml_response_str);

            if (!ml_response || !ml_response.has("predicted_volume")){
                throw std::runtime_error("wrong ml response");
            }

            double predicted_volume = ml_response["predicted_volume"].d();

            Selected_trucks ans = select_trucks(predicted_volume, g_park, optimize_by_cost);

            response["predicted volume"] = predicted_volume;
            response["number_of_trucks"] = ans.selected_truck_ids.size();
            response["total_capacity"] = ans.tot_capacity;
            response["total_cost"] = ans.tot_cost;
            response["usage_percent"] = ans.usage_percent;

            response["selected_trucks"] = std::move(ans.selected_truck_ids);

            if (ans.tot_capacity < predicted_volume){
                response["message"] = "insufficient park capacity!";
            } else {
                response["message"] = "success";
            }

            return crow::response(200, response.dump());

        } catch (const std::runtime_error& e){
            response["status"] = "error";
            response["messge"] = e.what();
            return crow::response(500, response.dump());
        }

        return crow::response(500, "{\"status\":\"error\"}");
    } );

    std::cout << "starting server" << std::endl;
    app.port(8080).multithreaded().run();

    return 0;
}