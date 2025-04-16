import os
import datetime
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, abort
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from urllib.parse import urlparse
from models import db, User, URLCheckHistory
from forms import LoginForm, RegistrationForm, UserSearchForm, UserEditForm, URLCheckForm, ChangePasswordForm
import dl_web  # Import original application code

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'phishing-detection-secret-key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    
    # Initialize login manager
    login_manager = LoginManager()
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please login to access this page.'
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
    # Create database tables
    with app.app_context():
        db.create_all()
        # Ensure there is at least one admin account
        if not User.query.filter_by(is_admin=True).first():
            admin = User(username='admin', email='admin@example.com', is_admin=True)
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
    
    # Home page (website homepage)
    @app.route('/home')
    def home():
        return render_template('home.html', title='Phishing URL Detection System - Home')
    
    # Redirect root URL to home page
    @app.route('/')
    def root():
        return redirect(url_for('home'))
    
    # Login page
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(username=form.username.data).first()
            if user is None or not user.check_password(form.password.data):
                flash('Incorrect username or password')
                return redirect(url_for('login'))
            login_user(user, remember=form.remember_me.data)
            user.last_login = datetime.datetime.utcnow()
            db.session.commit()
            next_page = request.args.get('next')
            if not next_page or urlparse(next_page).netloc != '':
                next_page = url_for('index')
            return redirect(next_page)
        return render_template('login.html', title='Login', form=form)
    
    # Registration page
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        form = RegistrationForm()
        if form.validate_on_submit():
            user = User(username=form.username.data, email=form.email.data)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Congratulations, you have successfully registered!')
            return redirect(url_for('login'))
        return render_template('register.html', title='Register', form=form)
    
    # Logout
    @app.route('/logout')
    def logout():
        logout_user()
        return redirect(url_for('login'))  # Changed to redirect to login page
    
    # User profile page
    @app.route('/profile')
    @login_required
    def profile():
        history = URLCheckHistory.query.filter_by(user_id=current_user.id).order_by(URLCheckHistory.check_time.desc()).limit(10).all()
        return render_template('profile.html', title='Profile', user=current_user, history=history)
    
    # Change password
    @app.route('/change_password', methods=['GET', 'POST'])
    @login_required
    def change_password():
        form = ChangePasswordForm()
        if form.validate_on_submit():
            if not current_user.check_password(form.current_password.data):
                flash('Current password is incorrect')
                return redirect(url_for('change_password'))
            current_user.set_password(form.new_password.data)
            db.session.commit()
            flash('Password has been successfully changed')
            return redirect(url_for('profile'))
        return render_template('change_password.html', title='Change Password', form=form)
    
    # Admin user management
    @app.route('/admin/users', methods=['GET', 'POST'])
    @login_required
    def admin_users():
        if not current_user.is_admin:
            abort(403)
        
        form = UserSearchForm()
        query = User.query
        
        if form.validate_on_submit():
            search_term = form.search.data
            query = query.filter(
                (User.username.like(f'%{search_term}%')) | 
                (User.email.like(f'%{search_term}%'))
            )
        
        users = query.order_by(User.id).all()
        return render_template('admin_users.html', title='User Management', form=form, users=users)
    
    # Admin edit user
    @app.route('/admin/user/<int:id>', methods=['GET', 'POST'])
    @login_required
    def admin_edit_user(id):
        if not current_user.is_admin:
            abort(403)
        
        user = User.query.get_or_404(id)
        form = UserEditForm(obj=user)
        
        if form.validate_on_submit():
            # Prevent the last admin from removing their own admin privileges
            if user.is_admin and not form.is_admin.data:
                admin_count = User.query.filter_by(is_admin=True).count()
                if admin_count <= 1:
                    flash('You cannot remove the privileges of the last admin account')
                    return redirect(url_for('admin_edit_user', id=id))
            
            user.username = form.username.data
            user.email = form.email.data
            user.is_admin = form.is_admin.data
            db.session.commit()
            flash(f'User {user.username} information has been updated')
            return redirect(url_for('admin_users'))
            
        return render_template('admin_edit_user.html', title='Edit User', form=form, user=user)
    
    # Admin delete user
    @app.route('/admin/user/delete/<int:id>', methods=['POST'])
    @login_required
    def admin_delete_user(id):
        if not current_user.is_admin:
            abort(403)
        
        user = User.query.get_or_404(id)
        
        # Prevent deleting yourself
        if user.id == current_user.id:
            flash('You cannot delete your own account')
            return redirect(url_for('admin_users'))
        
        # Prevent deleting the last admin
        if user.is_admin:
            admin_count = User.query.filter_by(is_admin=True).count()
            if admin_count <= 1:
                flash('You cannot delete the last admin account')
                return redirect(url_for('admin_users'))
        
        username = user.username
        db.session.delete(user)
        db.session.commit()
        flash(f'User {username} has been deleted')
        return redirect(url_for('admin_users'))
    
    # Admin view detection history
    @app.route('/admin/history')
    @login_required
    def admin_history():
        if not current_user.is_admin:
            abort(403)
        
        history = URLCheckHistory.query.order_by(URLCheckHistory.check_time.desc()).all()
        return render_template('admin_history.html', title='Detection History', history=history)
    
    # Main page (URL detection)
    @app.route('/detection', methods=['GET', 'POST'])
    @login_required  # Add login requirement
    def index():
        form = URLCheckForm()
        # Get available model options
        available_models = ["demo_mode"] if dl_web.args.demo else dl_web.model_paths.keys()
        form.model.choices = [(model, model) for model in available_models]
        
        if form.validate_on_submit():
            model_name = form.model.data
            batch_results = []  # Used to store batch detection results
            urls_to_check = []  # List of URLs to check
            
            # 获取当前活动选项卡
            active_tab = form.active_tab.data or 'single'  # 默认为单URL选项卡
            
            # 根据活动选项卡处理输入
            if active_tab == 'single':
                # 只处理单个URL
                if form.url.data:
                    urls_to_check.append(form.url.data)
            elif active_tab == 'bulk':
                # 只处理批量文本输入
                if form.bulk_urls.data:
                    bulk_urls = form.bulk_urls.data.strip().split('\n')
                    urls_to_check.extend([url.strip() for url in bulk_urls if url.strip()])
            elif active_tab == 'file':
                # 只处理文件上传
                if form.url_file.data and form.url_file.data.filename:
                    try:
                        file_content = form.url_file.data.read().decode('utf-8')
                        file_urls = file_content.strip().split('\n')
                        urls_to_check.extend([url.strip() for url in file_urls if url.strip()])
                    except Exception as e:
                        flash(f'Error reading file: {str(e)}')
                        return redirect(url_for('index'))
            
            # 去除重复URL
            urls_to_check = list(dict.fromkeys(urls_to_check))
            
            # 如果没有有效URL，返回错误
            if not urls_to_check:
                flash('No valid URLs provided for detection')
                return redirect(url_for('index'))
            
            # Prepare model
            if model_name == "demo_mode" or dl_web.args.demo:
                phishing_url_test = dl_web.PhishingUrlTest()
            else:
                model_path = dl_web.model_paths.get(model_name, None)
                if not model_path:
                    flash('Model not found')
                    return redirect(url_for('index'))
                
                phishing_url_test = dl_web.PhishingUrlTest(model_path=model_path)
                phishing_url_test.load_model()
            
            # Perform batch detection
            for url in urls_to_check:
                try:
                    result, confidence = phishing_url_test.classify_url(url)
                    
                    # Save detection history
                    history = URLCheckHistory(
                        url=url,
                        result=result,
                        confidence=float(confidence),
                        model_used=model_name,
                        user_id=current_user.id
                    )
                    db.session.add(history)
                    
                    batch_results.append({
                        'url': url,
                        'result': result,
                        'confidence': confidence
                    })
                except Exception as e:
                    # Handle detection failure for URL
                    batch_results.append({
                        'url': url,
                        'result': 'Detection failed',
                        'confidence': 0,
                        'error': str(e)
                    })
            
            # Submit all history records
            db.session.commit()
            
            # 根据选项卡类型和检测结果数量确定返回模式
            if active_tab == 'single' and len(batch_results) == 1:
                # 单URL检测结果
                result_data = batch_results[0]
                return render_template('index.html', title='Phishing URL Detection', form=form, 
                                   prediction=result_data['result'], 
                                   confidence=result_data['confidence'], 
                                   url=result_data['url'], 
                                   model_name=model_name,
                                   active_tab=active_tab)
            else:
                # 批量检测结果 - 适用于批量URL或文件上传
                return render_template('index.html', title='Phishing URL Detection', form=form, 
                                   batch_results=batch_results,
                                   model_name=model_name,
                                   active_tab=active_tab)
        
        return render_template('index.html', title='Phishing URL Detection', form=form)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8001, debug=True) 