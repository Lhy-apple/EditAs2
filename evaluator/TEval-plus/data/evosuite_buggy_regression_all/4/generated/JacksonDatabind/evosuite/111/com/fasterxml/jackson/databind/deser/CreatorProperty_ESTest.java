/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:48:03 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationCollector;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.io.PipedInputStream;
import java.lang.annotation.Annotation;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CreatorProperty_ESTest extends CreatorProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(false);
      Integer integer0 = new Integer(1661);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "'; inject id '");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2920), object0, propertyMetadata0);
      try { 
        creatorProperty0.setAndReturn(object0, creatorProperty0);
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
      PropertyName propertyName0 = PropertyName.construct("au9g)yAl");
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(true);
      Integer integer0 = new Integer((-1163));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "au9g)yAl", integer0, "c3|jO]7bv]ss})(f");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-1163), object0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withSimpleName("c3|jO]7bv]ss})(f");
      assertEquals("c3|jO]7bv]ss})(f", settableBeanProperty0.getName());
      assertEquals((-1163), settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(false);
      Integer integer0 = new Integer(1661);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "'; inject id '");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2920), object0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withNullProvider((NullValueProvider) null);
      assertEquals((-2920), settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.construct("'; inject id '");
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(true);
      Integer integer0 = new Integer((-2920));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "Cannot deserialize a Map.Entry out of empty JSON Object");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-1163), object0, propertyMetadata0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        creatorProperty0.deserializeSetAndReturn((JsonParser) null, defaultDeserializationContext_Impl0, pipedInputStream0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property ''; inject id ''
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(false);
      Integer integer0 = new Integer(1661);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "'; inject id '");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2920), object0, propertyMetadata0);
      // Undeclared exception!
      try { 
        creatorProperty0.getDeclaringClass();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.SettableBeanProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-1163), (Object) null, propertyMetadata0);
      creatorProperty0.markAsIgnorable();
      assertTrue(creatorProperty0.isIgnorable());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.construct("", "");
      PlaceholderForType placeholderForType0 = new PlaceholderForType((-2998));
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(placeholderForType0, (TypeFactory) null);
      Class<Object> class0 = Object.class;
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(placeholderForType0, classNameIdResolver0, "", true, placeholderForType0);
      Class<Integer> class1 = Integer.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, placeholderForType0, propertyName0, asExternalTypeDeserializer0, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, (-1611), propertyName0, propertyMetadata0);
      Object object0 = creatorProperty0.getInjectableValueId();
      assertEquals((-1611), creatorProperty0.getCreatorIndex());
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.construct("'; inject id '");
      JavaType javaType0 = TypeFactory.unknownType();
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(javaType0, (TypeIdResolver) null, "'; inject id '", false, javaType0);
      Class<Integer> class0 = Integer.class;
      PlaceholderForType placeholderForType0 = new PlaceholderForType((-874));
      Class<Object> class1 = Object.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class0, (Annotation) null, class1, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, placeholderForType0, (PropertyName) null, asExternalTypeDeserializer0, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, (-1163), objectIdGenerators_IntSequenceGenerator0, propertyMetadata0);
      creatorProperty0.toString();
      assertEquals((-1163), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.valueOf(false);
      Integer integer0 = new Integer((-2920));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, (String) null);
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2920), (Object) null, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        creatorProperty0.inject(defaultDeserializationContext_Impl0, object0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.construct("'; inject id '");
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(true);
      Integer integer0 = new Integer((-2920));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "Cannot deserialize a Map.Entry out of empty JSON Object");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-1163), object0, propertyMetadata0);
      creatorProperty0.setFallbackSetter(creatorProperty0);
      assertEquals((-1163), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.construct("'; inject id '");
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(true);
      Integer integer0 = new Integer((-2920));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "Cannot deserialize a Map.Entry out of empty JSON Object");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-1163), object0, propertyMetadata0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      byte[] byteArray0 = new byte[6];
      JsonParser jsonParser0 = jsonFactory0.createParser(byteArray0);
      try { 
        creatorProperty0.deserializeAndSet(jsonParser0, defaultDeserializationContext_Impl0, jsonFactory0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property ''; inject id ''
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(false);
      Integer integer0 = new Integer(1661);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "'; inject id '");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2920), object0, propertyMetadata0);
      int int0 = creatorProperty0.getCreatorIndex();
      assertEquals((-2920), int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.construct("'; inject id '");
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(true);
      Integer integer0 = new Integer((-2920));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "Cannot deserialize a Map.Entry out of empty JSON Object");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-1163), object0, propertyMetadata0);
      creatorProperty0.isIgnorable();
      assertEquals((-1163), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(false);
      Integer integer0 = new Integer(1661);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "'; inject id '");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2920), object0, propertyMetadata0);
      CreatorProperty creatorProperty1 = (CreatorProperty)creatorProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertNotSame(creatorProperty1, creatorProperty0);
      
      creatorProperty1._fallbackSetter = (SettableBeanProperty) creatorProperty0;
      creatorProperty1.fixAccess((DeserializationConfig) null);
      assertEquals((-2920), creatorProperty1.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(false);
      Integer integer0 = new Integer(1661);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "'; inject id '");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2920), object0, propertyMetadata0);
      creatorProperty0.fixAccess((DeserializationConfig) null);
      assertEquals((-2920), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Boolean boolean0 = Boolean.valueOf(false);
      Integer integer0 = new Integer((-2920));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, (String) null);
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2920), object0, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        creatorProperty0.inject(defaultDeserializationContext_Impl0, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(false);
      Integer integer0 = new Integer(1661);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "'; inject id '");
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2920), object0, propertyMetadata0);
      Class<Annotation> class0 = Annotation.class;
      creatorProperty0.getAnnotation(class0);
      assertEquals((-2920), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.construct("'; inject id '");
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      Boolean boolean0 = Boolean.valueOf(false);
      Integer integer0 = new Integer(1661);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "'; inject id '", integer0, "'; inject id '");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2920), object0, propertyMetadata0);
      creatorProperty0._fallbackSetter = (SettableBeanProperty) creatorProperty0;
      // Undeclared exception!
      creatorProperty0.set("'; inject id '", annotationMap0);
  }
}