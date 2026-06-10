def mGAN(n=500, z_dim=1, simulation='type1error', batch_size=64, epochs_num=1000,
         nstd=1.0, z_dist='gaussian', x_dims=1, y_dims=1, a_x=0.05, M=500, k=2, boot_num=1000,
         noise_dimension=10, hidden_layer_size=512, normal_ini=False, preprocess='normalize',
         G_lr=1e-5, using_orcale=False, lambda_1=1, lambda_2=1, using_Gen='1',
         boor_rv_type="gaussian", wgt_decay=0, lambda_3=1, drop_out_p=0.2, noise_dimension_var=1,
         noise_dimension_type="normal", lambda_4=1):
    """
    Corrected mGAN for oracle and non-oracle runs.

    Main fixes:
    1. Do not call G_zx.eval() / G_zy.eval() when using_orcale=True.
    2. Use pairwise-distance median heuristic for sigma_u and sigma_v.
    3. Generate oracle fake samples directly from P(X|Z) and P(Y|Z), independently.
    4. Use add-one correction for the bootstrap p-value.
    """
    if simulation == 'type1error':
        sim_x, sim_y, sim_z = generate_samples_random(
            size=n, sType='H0', dx=x_dims, dy=y_dims, dz=z_dim,
            nstd=nstd, alpha_x=a_x, dist_z=z_dist, preprocess=preprocess)
    elif simulation == 'power':
        sim_x, sim_y, sim_z = generate_samples_random(
            size=n, sType='H1', dx=x_dims, dy=y_dims, dz=z_dim,
            nstd=nstd, alpha_x=a_x, dist_z=z_dist, preprocess=preprocess)
    else:
        raise ValueError('Test does not exist.')

    x, y, z = sim_x, sim_y, sim_z

    # Correct median heuristic: use distances, not kernel values.
    w_dist = torch.linalg.vector_norm(
        z.repeat(n, 1, 1) - torch.swapaxes(z.repeat(n, 1, 1), 0, 1),
        ord=1, dim=2)
    sigma_w = torch.median(w_dist[w_dist > 0]).clamp_min(1e-8).item()

    u_dist = torch.linalg.vector_norm(
        y.repeat(n, 1, 1) - torch.swapaxes(y.repeat(n, 1, 1), 0, 1),
        ord=1, dim=2)
    sigma_u = torch.median(u_dist[u_dist > 0]).clamp_min(1e-8).item()

    v_dist = torch.linalg.vector_norm(
        x.repeat(n, 1, 1) - torch.swapaxes(x.repeat(n, 1, 1), 0, 1),
        ord=1, dim=2)
    sigma_v = torch.median(v_dist[v_dist > 0]).clamp_min(1e-8).item()

    test_size = int(n / k)
    stat_all = torch.zeros(k, 1)
    boot_temp_all = torch.zeros(k, boot_num)
    cur_k = 0

    for k_fold in range(k):
        k_fold_start = int(n / k * k_fold)
        k_fold_end = int(n / k * (k_fold + 1))

        X_test = x[k_fold_start:k_fold_end]
        Y_test = y[k_fold_start:k_fold_end]
        Z_test = z[k_fold_start:k_fold_end]

        X_train = torch.cat((x[0:k_fold_start], x[k_fold_end:]))
        Y_train = torch.cat((y[0:k_fold_start], y[k_fold_end:]))
        Z_train = torch.cat((z[0:k_fold_start], z[k_fold_end:]))

        if k == 1:
            X_train, Y_train, Z_train = X_test, Y_test, Z_test

        if not using_orcale:
            train_xyz = DatasetSelect_GAN(X_train, Y_train, Z_train, batch_size)
            DataLoader_xyz = torch.utils.data.DataLoader(train_xyz, batch_size=batch_size, shuffle=True)

            # train_ver3 creates a fresh generator internally in this code version.
            G_zy, G_zx = train_ver3(
                X=X_train, Y=Y_train, Z=Z_train, M=M,
                X_test=X_test, Y_test=Y_test, Z_test=Z_test,
                noise_dimension=noise_dimension, noise_type=noise_dimension_type,
                G_lr=G_lr, hidden_layer_size=hidden_layer_size,
                DataLoader=DataLoader_xyz, BN_type=False, ReLU_coef=0.1,
                epochs_num=epochs_num, sigma_z=sigma_w, sigma_x=sigma_v, sigma_y=sigma_u,
                normal_ini=normal_ini, lambda_1=lambda_1, lambda_2=lambda_2,
                using_Gen=using_Gen, wgt_decay=wgt_decay, lambda_3=lambda_3,
                drop_out_p=drop_out_p, noise_dimension_var=noise_dimension_var,
                lambda_4=lambda_4)

            G_zx = G_zx.eval()
            G_zy = G_zy.eval()

        dataset_test = DatasetSelect(X_test, Y_test, Z_test)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

        gen_x_all = torch.zeros(test_size, M)
        gen_y_all = torch.zeros(test_size, M)
        z_all = torch.zeros(test_size, z_dim)
        x_all = torch.zeros(test_size, x_dims)
        y_all = torch.zeros(test_size, y_dims)

        cur_itr = 0
        for i, (x_test, y_test, z_test) in enumerate(dataloader_test):
            z_test_temp = z_test.repeat(M, 1)

            if not using_orcale:
                z_test_temp_device = z_test_temp.to(device)
                Noise_fake = sample_noise(z_test_temp_device.size(0), noise_dimension, "normal").to(device)
                with torch.no_grad():
                    fake_x = G_zx(torch.cat((z_test_temp_device, Noise_fake), dim=1)).reshape(1, -1).cpu()

                Noise_fake = sample_noise(z_test_temp_device.size(0), noise_dimension, "normal").to(device)
                with torch.no_grad():
                    fake_y = G_zy(torch.cat((z_test_temp_device, Noise_fake), dim=1)).reshape(1, -1).cpu()

            else:
                # Oracle: always sample independent fake X and fake Y from marginal conditionals.
                fake_x, fake_y = generate_samples_from_fixed_Z_random(
                    z_test_temp, size=M, sType='H0', dx=x_dims, dy=y_dims,
                    dz=z_dim, nstd=nstd, alpha_x=a_x, dist_z=z_dist)

            gen_x_all[cur_itr, :] = fake_x.detach().reshape(-1).cpu()
            gen_y_all[cur_itr, :] = fake_y.detach().reshape(-1).cpu()
            x_all[cur_itr, :] = x_test
            y_all[cur_itr, :] = y_test
            z_all[cur_itr, :] = z_test
            cur_itr += 1

        cur_stat, cur_boot_temp = get_p_value_stat_1(
            boot_num, M, test_size,
            gen_x_all.to(device), gen_y_all.to(device),
            x_all.to(device), y_all.to(device), z_all.to(device),
            sigma_w, sigma_u, sigma_v, boor_rv_type)

        stat_all[cur_k, :] = cur_stat
        boot_temp_all[cur_k, :] = torch.from_numpy(cur_boot_temp)
        cur_k += 1

    boot_mean = torch.mean(boot_temp_all, dim=0).numpy()
    stat_mean = torch.mean(stat_all).item()

    return (1.0 + np.sum(boot_mean > stat_mean)) / (boot_num + 1.0)
