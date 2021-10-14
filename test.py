def test_model_input_plot():
    import config
    import torch
    from yolodwg import DwgKeyPointsModel, DwgKeyPointsResNet50, non_zero_loss
    from plot import plot_batch_grid

    #model = DwgKeyPointsModel(max_points=max_points, num_coordinates=num_coordinates).to(config.device)

    checkpoint = torch.load('runs/2/best.weights', map_location=config.device)
    max_points = checkpoint['max_points']
    num_pnt_classes = checkpoint['num_pnt_classes']

    model = DwgKeyPointsModel(max_points=max_points, num_pnt_classes=num_pnt_classes)
    #model = DwgKeyPointsResNet50(requires_grad=True, max_points=max_points)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()

    res = None

    for i in range(100):
        imgs = torch.rand([4, 3, 128, 128])
        imgs = imgs.to(config.device)
        out = model(imgs)
        if res is None:
            res = out
        else:
            res = torch.cat([res, out])
        #plot_batch_grid(
        #        input_images=imgs,
        #        true_keypoints=None,
        #        predictions=out,
        #        plot_save_file=f'runs/debug_{i}.png')

    print(torch.min(res).item())
    print(torch.mean(res).item())
    print(torch.max(res).item(), '\n')

if __name__ == "__main__":
    test_model_input_plot()