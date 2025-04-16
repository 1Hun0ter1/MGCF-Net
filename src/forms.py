from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, Optional
from models import User

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField(
        'Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('This username is already in use. Please choose another username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('This email is already registered. Please use another email or recover your password.')

class UserSearchForm(FlaskForm):
    search = StringField('Search User', validators=[DataRequired()])
    submit = SubmitField('Search')

class UserEditForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    is_admin = BooleanField('Admin Privileges')
    submit = SubmitField('Save')

class URLCheckForm(FlaskForm):
    url = StringField('URL', validators=[Optional()])
    bulk_urls = TextAreaField('Batch URL Detection', validators=[Optional()], 
                           description='Enter one URL per line for batch detection')
    url_file = FileField('Upload URL File', validators=[
        Optional(),
        FileAllowed(['txt'], 'Only txt files are allowed!')
    ], description='Upload a txt file containing URLs, one URL per line')
    model = SelectField('Detection Model', choices=[('demo_mode', 'Demo Mode')])
    active_tab = StringField('Active Tab', validators=[Optional()])
    submit = SubmitField('Detect')
    
    def validate(self, extra_validators=None):
        if not super(URLCheckForm, self).validate():
            return False
        
        # 基于活动选项卡进行验证
        active_tab = self.active_tab.data or 'single'  # 默认为单URL选项卡
        
        if active_tab == 'single' and not self.url.data:
            self.url.errors.append('Please provide a URL to detect')
            return False
        elif active_tab == 'bulk' and not self.bulk_urls.data:
            self.bulk_urls.errors.append('Please provide URLs for batch detection')
            return False
        elif active_tab == 'file' and not (self.url_file.data and self.url_file.data.filename):
            self.url_file.errors.append('Please upload a file with URLs')
            return False
            
        return True

class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField(
        'Confirm New Password', validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Change Password') 