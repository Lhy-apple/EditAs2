/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:45:53 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.BeanPropertyMap;
import com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler;
import com.fasterxml.jackson.databind.deser.impl.PropertyBasedCreator;
import com.fasterxml.jackson.databind.deser.impl.PropertyValueBuffer;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationCollector;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.ByteArrayInputStream;
import java.lang.annotation.Annotation;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ExternalTypeHandler_ESTest extends ExternalTypeHandler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder(javaType0);
      PropertyName propertyName0 = PropertyName.construct(":ax.WH]?rL^/", ":ax.WH]?rL^/");
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, typeFactory0);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, ":ax.WH]?rL^/", true, javaType0);
      Class<ByteArrayInputStream> class0 = ByteArrayInputStream.class;
      AnnotationCollector.OneAnnotation annotationCollector_OneAnnotation0 = new AnnotationCollector.OneAnnotation(class0, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf(true);
      Integer integer0 = new Integer(1896);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, ":ax.WH]?rL^/", integer0, ":ax.WH]?rL^/");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asArrayTypeDeserializer0, annotationCollector_OneAnnotation0, (AnnotatedParameter) null, 1896, annotationCollector_OneAnnotation0, propertyMetadata0);
      externalTypeHandler_Builder0.addExternal(creatorProperty0, asArrayTypeDeserializer0);
      HashSet<SettableBeanProperty> hashSet0 = new HashSet<SettableBeanProperty>();
      HashMap<String, List<PropertyName>> hashMap0 = new HashMap<String, List<PropertyName>>();
      BeanPropertyMap beanPropertyMap0 = BeanPropertyMap.construct((Collection<SettableBeanProperty>) hashSet0, true, (Map<String, List<PropertyName>>) hashMap0);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build(beanPropertyMap0);
      // Undeclared exception!
      try { 
        externalTypeHandler0.handlePropertyValue((JsonParser) null, (DeserializationContext) null, ":ax.WH]?rL^/", classNameIdResolver0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder(javaType0);
      PropertyName propertyName0 = PropertyName.construct(":ax.WH]?rL^/", ":ax.WH]?rL^/");
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, typeFactory0);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, ":ax.WH]?rL^/", true, javaType0);
      Class<ByteArrayInputStream> class0 = ByteArrayInputStream.class;
      AnnotationCollector.OneAnnotation annotationCollector_OneAnnotation0 = new AnnotationCollector.OneAnnotation(class0, (Annotation) null);
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-23));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, ":ax.WH]?rL^/", integer0, ":ax.WH]?rL^/");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asArrayTypeDeserializer0, annotationCollector_OneAnnotation0, (AnnotatedParameter) null, 1896, annotationCollector_OneAnnotation0, propertyMetadata0);
      externalTypeHandler_Builder0.addExternal(creatorProperty0, asArrayTypeDeserializer0);
      HashSet<SettableBeanProperty> hashSet0 = new HashSet<SettableBeanProperty>();
      hashSet0.add(creatorProperty0);
      HashMap<String, List<PropertyName>> hashMap0 = new HashMap<String, List<PropertyName>>();
      BeanPropertyMap beanPropertyMap0 = BeanPropertyMap.construct((Collection<SettableBeanProperty>) hashSet0, true, (Map<String, List<PropertyName>>) hashMap0);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build(beanPropertyMap0);
      assertNotNull(externalTypeHandler0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder(javaType0);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build((BeanPropertyMap) null);
      ExternalTypeHandler externalTypeHandler1 = externalTypeHandler0.start();
      assertNotSame(externalTypeHandler1, externalTypeHandler0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = ExternalTypeHandler.builder((JavaType) null);
      assertNotNull(externalTypeHandler_Builder0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder(javaType0);
      PropertyName propertyName0 = PropertyName.construct(":ax.WH]?rL^/", ":ax.WH]?rL^/");
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, typeFactory0);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, ":ax.WH]?rL^/", true, javaType0);
      Class<ByteArrayInputStream> class0 = ByteArrayInputStream.class;
      AnnotationCollector.OneAnnotation annotationCollector_OneAnnotation0 = new AnnotationCollector.OneAnnotation(class0, (Annotation) null);
      Boolean boolean0 = Boolean.TRUE;
      Integer integer0 = new Integer((-23));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, ":ax.WH]?rL^/", integer0, ":ax.WH]?rL^/");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asArrayTypeDeserializer0, annotationCollector_OneAnnotation0, (AnnotatedParameter) null, 1896, annotationCollector_OneAnnotation0, propertyMetadata0);
      externalTypeHandler_Builder0.addExternal(creatorProperty0, asArrayTypeDeserializer0);
      HashSet<SettableBeanProperty> hashSet0 = new HashSet<SettableBeanProperty>();
      HashMap<String, List<PropertyName>> hashMap0 = new HashMap<String, List<PropertyName>>();
      BeanPropertyMap beanPropertyMap0 = BeanPropertyMap.construct((Collection<SettableBeanProperty>) hashSet0, true, (Map<String, List<PropertyName>>) hashMap0);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build(beanPropertyMap0);
      // Undeclared exception!
      try { 
        externalTypeHandler0.handleTypePropertyValue((JsonParser) null, (DeserializationContext) null, ":ax.WH]?rL^/", classNameIdResolver0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder(javaType0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build((BeanPropertyMap) null);
      boolean boolean0 = externalTypeHandler0.handleTypePropertyValue((JsonParser) null, defaultDeserializationContext_Impl0, "TG+eoF20]", externalTypeHandler_Builder0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder(javaType0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build((BeanPropertyMap) null);
      boolean boolean0 = externalTypeHandler0.handlePropertyValue((JsonParser) null, defaultDeserializationContext_Impl0, "", javaType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder(javaType0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, typeFactory0);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, ":ax.WH]?rL^/", true, javaType0);
      Class<ByteArrayInputStream> class0 = ByteArrayInputStream.class;
      AnnotationCollector.OneAnnotation annotationCollector_OneAnnotation0 = new AnnotationCollector.OneAnnotation(class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asArrayTypeDeserializer0, annotationCollector_OneAnnotation0, (AnnotatedParameter) null, 2, annotationCollector_OneAnnotation0, propertyMetadata0);
      externalTypeHandler_Builder0.addExternal(creatorProperty0, asArrayTypeDeserializer0);
      HashSet<SettableBeanProperty> hashSet0 = new HashSet<SettableBeanProperty>();
      HashMap<String, List<PropertyName>> hashMap0 = new HashMap<String, List<PropertyName>>();
      BeanPropertyMap beanPropertyMap0 = BeanPropertyMap.construct((Collection<SettableBeanProperty>) hashSet0, true, (Map<String, List<PropertyName>>) hashMap0);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build(beanPropertyMap0);
      ExternalTypeHandler externalTypeHandler1 = new ExternalTypeHandler(externalTypeHandler0);
      JsonFactory jsonFactory0 = new JsonFactory((ObjectCodec) null);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Object object0 = externalTypeHandler1.complete(jsonParser0, deserializationContext0, (Object) externalTypeHandler_Builder0);
      assertSame(externalTypeHandler_Builder0, object0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder(javaType0);
      PropertyName propertyName0 = PropertyName.construct(":ax.WH]?rL^/", ":ax.WH]?rL^/");
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, typeFactory0);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, ":ax.WH]?rL^/", true, javaType0);
      Class<ByteArrayInputStream> class0 = ByteArrayInputStream.class;
      AnnotationCollector.OneAnnotation annotationCollector_OneAnnotation0 = new AnnotationCollector.OneAnnotation(class0, (Annotation) null);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asArrayTypeDeserializer0, annotationCollector_OneAnnotation0, (AnnotatedParameter) null, 1896, annotationCollector_OneAnnotation0, propertyMetadata0);
      externalTypeHandler_Builder0.addExternal(creatorProperty0, asArrayTypeDeserializer0);
      HashSet<SettableBeanProperty> hashSet0 = new HashSet<SettableBeanProperty>();
      HashMap<String, List<PropertyName>> hashMap0 = new HashMap<String, List<PropertyName>>();
      BeanPropertyMap beanPropertyMap0 = BeanPropertyMap.construct((Collection<SettableBeanProperty>) hashSet0, true, (Map<String, List<PropertyName>>) hashMap0);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build(beanPropertyMap0);
      ExternalTypeHandler externalTypeHandler1 = new ExternalTypeHandler(externalTypeHandler0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        externalTypeHandler1.complete((JsonParser) null, (DeserializationContext) defaultDeserializationContext_Impl0, (PropertyValueBuffer) null, (PropertyBasedCreator) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder(javaType0);
      PropertyName propertyName0 = PropertyName.construct(":ax.WH]?rL^/", ":ax.WH]?rL^/");
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, typeFactory0);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, ":ax.WH]?rL^/", true, javaType0);
      Class<ByteArrayInputStream> class0 = ByteArrayInputStream.class;
      AnnotationCollector.OneAnnotation annotationCollector_OneAnnotation0 = new AnnotationCollector.OneAnnotation(class0, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf(true);
      Integer integer0 = new Integer(1896);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, ":ax.WH]?rL^/", integer0, ":ax.WH]?rL^/");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asArrayTypeDeserializer0, annotationCollector_OneAnnotation0, (AnnotatedParameter) null, 1896, annotationCollector_OneAnnotation0, propertyMetadata0);
      externalTypeHandler_Builder0.addExternal(creatorProperty0, asArrayTypeDeserializer0);
      externalTypeHandler_Builder0.addExternal(creatorProperty0, asArrayTypeDeserializer0);
      assertEquals(":ax.WH]?rL^/", creatorProperty0.getName());
  }
}
