/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:05:50 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BasicDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.lang.annotation.Annotation;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CreatorProperty_ESTest extends CreatorProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      Class<Annotation> class0 = Annotation.class;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 10, class0, propertyMetadata0);
      try { 
        creatorProperty0.setAndReturn(annotationMap0, propertyMetadata0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property ''
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Annotation> class0 = Annotation.class;
      AnnotationMap annotationMap0 = AnnotationMap.of(class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 5582, object0, propertyMetadata0);
      JsonDeserializer<ObjectIdResolver> jsonDeserializer0 = (JsonDeserializer<ObjectIdResolver>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      CreatorProperty creatorProperty1 = new CreatorProperty(creatorProperty0, jsonDeserializer0, jsonDeserializer0);
      JsonDeserializer<Object> jsonDeserializer1 = SettableBeanProperty.MISSING_VALUE_DESERIALIZER;
      SettableBeanProperty settableBeanProperty0 = creatorProperty1.withValueDeserializer(jsonDeserializer1);
      assertEquals(5582, creatorProperty1.getCreatorIndex());
      assertFalse(settableBeanProperty0.hasValueDeserializer());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Annotation> class0 = Annotation.class;
      AnnotationMap annotationMap0 = AnnotationMap.of(class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 5582, object0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withName((PropertyName) null);
      assertEquals(5582, settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<JsonEncoding> class0 = JsonEncoding.class;
      AnnotationMap annotationMap0 = AnnotationMap.of(class0, (Annotation) null);
      Class<Integer> class1 = Integer.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class1);
      PropertyName propertyName0 = BasicDeserializerFactory.UNWRAPPED_CREATOR_PARAM_NAME;
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 1, valueInstantiator_Base0, propertyMetadata0);
      try { 
        creatorProperty0.set(valueInstantiator_Base0, annotationMap0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property '@JsonUnwrapped'
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<JsonEncoding> class0 = JsonEncoding.class;
      AnnotationMap annotationMap0 = AnnotationMap.of(class0, (Annotation) null);
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      PropertyMetadata propertyMetadata0 = beanProperty_Bogus0.getMetadata();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 2173, (Object) null, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withNullProvider((NullValueProvider) null);
      assertEquals(2173, settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      PropertyName propertyName0 = BasicDeserializerFactory.UNWRAPPED_CREATOR_PARAM_NAME;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 0, (Object) null, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory();
      char[] charArray0 = new char[2];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        creatorProperty0.deserializeSetAndReturn(jsonParser0, defaultDeserializationContext_Impl0, (Object) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property ''
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<JsonEncoding> class0 = JsonEncoding.class;
      AnnotationMap annotationMap0 = AnnotationMap.of(class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 2182, class0, propertyMetadata0);
      creatorProperty0.getMember();
      assertEquals(2182, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<JsonEncoding> class0 = JsonEncoding.class;
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      AnnotationMap annotationMap0 = AnnotationMap.of(class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-1049), beanProperty_Bogus0, propertyMetadata0);
      creatorProperty0.markAsIgnorable();
      assertTrue(creatorProperty0.isIgnorable());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<JsonEncoding> class0 = JsonEncoding.class;
      AnnotationMap annotationMap0 = AnnotationMap.of(class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 2182, class0, propertyMetadata0);
      Object object0 = creatorProperty0.getInjectableValueId();
      assertNotNull(object0);
      assertEquals(2182, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<JsonEncoding> class0 = JsonEncoding.class;
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 2182, class0, propertyMetadata0);
      String string0 = creatorProperty0.toString();
      assertEquals("[creator property, name ''; inject id 'class com.fasterxml.jackson.core.JsonEncoding']", string0);
      assertEquals(2182, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Annotation> class0 = Annotation.class;
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      AnnotationMap annotationMap0 = AnnotationMap.of(class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      JavaType javaType0 = beanProperty_Bogus0.getType();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 2182, (Object) null, propertyMetadata0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      // Undeclared exception!
      try { 
        creatorProperty0.inject(defaultDeserializationContext_Impl0, objectIdGenerators_IntSequenceGenerator0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 2182, propertyMetadata0, propertyMetadata0);
      creatorProperty0.setFallbackSetter(creatorProperty0);
      assertEquals(2182, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<JsonEncoding> class0 = JsonEncoding.class;
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      JsonFactory jsonFactory0 = new JsonFactory();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Annotation> class1 = Annotation.class;
      Class<SimpleObjectIdResolver> class2 = SimpleObjectIdResolver.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class1, class2, class0);
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, mapLikeType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 6, beanProperty_Bogus0, propertyMetadata0);
      char[] charArray0 = new char[5];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0, 67, 57);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        creatorProperty0.deserializeAndSet(jsonParser0, defaultDeserializationContext_Impl0, (Object) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property ''
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<JsonEncoding> class0 = JsonEncoding.class;
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 2182, class0, propertyMetadata0);
      int int0 = creatorProperty0.getCreatorIndex();
      assertEquals(2182, int0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<JsonEncoding> class0 = JsonEncoding.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      PropertyName propertyName0 = BasicDeserializerFactory.UNWRAPPED_CREATOR_PARAM_NAME;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 2182, valueInstantiator_Base0, propertyMetadata0);
      creatorProperty0.isIgnorable();
      assertEquals(2182, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Annotation> class0 = Annotation.class;
      AnnotationMap annotationMap0 = AnnotationMap.of(class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 5582, object0, propertyMetadata0);
      JsonDeserializer<Object> jsonDeserializer0 = SettableBeanProperty.MISSING_VALUE_DESERIALIZER;
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withValueDeserializer(jsonDeserializer0);
      assertEquals(5582, settableBeanProperty0.getCreatorIndex());
      assertSame(settableBeanProperty0, creatorProperty0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 0, (Object) null, propertyMetadata0);
      creatorProperty0.fixAccess((DeserializationConfig) null);
      assertEquals(0, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      PropertyName propertyName0 = BasicDeserializerFactory.UNWRAPPED_CREATOR_PARAM_NAME;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 2182, propertyMetadata0, propertyMetadata0);
      // Undeclared exception!
      try { 
        creatorProperty0.findInjectableValue((DeserializationContext) null, annotationMap0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.CreatorProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<Annotation> class0 = Annotation.class;
      AnnotationMap annotationMap0 = AnnotationMap.of(class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, (JavaType) null, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 5582, object0, propertyMetadata0);
      creatorProperty0.getAnnotation(class0);
      assertEquals(5582, creatorProperty0.getCreatorIndex());
  }
}
