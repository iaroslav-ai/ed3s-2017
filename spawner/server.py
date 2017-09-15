import json
import tornado.httpserver
import tornado.ioloop
import tornado.web
from cluster import ClusterManager, EC2
import json
import os

all_ips_json = 'ips.json'
user_ip_json = 'uip.json'

def map_user_to_ip(user):
    if not os.path.exists(user_ip_json):
        json.dump({}, open(user_ip_json, 'w'))

    umap = json.load(open(user_ip_json))

    ips = []

    for f in [all_ips_json]:
        subset_ips = json.load(open(all_ips_json))
        ips+= subset_ips

    if not user in umap:
        for ip in ips:
            if any([ip == v for k, v in umap.items()]):
                continue

            umap[user] = ip
            json.dump(umap, open(user_ip_json, 'w'))
            break

    return umap[user]

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')

class GetJHUBHandler(tornado.web.RequestHandler):
    def post(self):
        data = self.request.body.decode('utf-8')
        data = json.loads(data)
        email = data['email']

        try:
            ip = map_user_to_ip(email)
            self.write({'ip': ip, 'status': 'ok'})
        except KeyError as ex:
            self.write({'status': 'eoi'})


if __name__ == '__main__':
    ips = EC2().get_worker_ips()
    json.dump(ips, open('ips.json', 'w'))

    app = tornado.web.Application([
        tornado.web.url(r'/process', GetJHUBHandler),
        tornado.web.url(r'/', MainHandler),
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8888)
    print('Starting server on port 8888')
    tornado.ioloop.IOLoop.instance().start()
