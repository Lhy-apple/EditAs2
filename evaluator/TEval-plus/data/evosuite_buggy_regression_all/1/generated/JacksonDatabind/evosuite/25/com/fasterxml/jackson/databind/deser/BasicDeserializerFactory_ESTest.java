/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:34:00 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.ContextAttributes;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BeanDeserializerModifier;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerFactory;
import com.fasterxml.jackson.databind.deser.ValueInstantiators;
import com.fasterxml.jackson.databind.deser.std.StdKeyDeserializers;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.module.SimpleAbstractTypeResolver;
import com.fasterxml.jackson.databind.module.SimpleDeserializers;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import java.lang.reflect.Type;
import java.sql.SQLException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BasicDeserializerFactory_ESTest extends BasicDeserializerFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      SimpleDeserializers simpleDeserializers0 = new SimpleDeserializers();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withAdditionalDeserializers(simpleDeserializers0);
      assertNotSame(beanDeserializerFactory0, deserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.withDeserializerModifier((BeanDeserializerModifier) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not pass null modifier
         //
         verifyException("com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ContextAttributes contextAttributes0 = ContextAttributes.getEmpty();
      ObjectReader objectReader0 = objectMapper0.reader(contextAttributes0);
      Class<SQLException> class0 = SQLException.class;
      ObjectReader objectReader1 = objectReader0.forType(class0);
      assertNotSame(objectReader0, objectReader1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DeserializerFactoryConfig deserializerFactoryConfig0 = beanDeserializerFactory0.getFactoryConfig();
      assertFalse(deserializerFactoryConfig0.hasDeserializerModifiers());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      ValueInstantiators.Base valueInstantiators_Base0 = new ValueInstantiators.Base();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withValueInstantiators(valueInstantiators_Base0);
      assertNotSame(beanDeserializerFactory0, deserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      StdKeyDeserializers stdKeyDeserializers0 = new StdKeyDeserializers();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withAdditionalKeyDeserializers(stdKeyDeserializers0);
      assertNotSame(beanDeserializerFactory0, deserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      SimpleAbstractTypeResolver simpleAbstractTypeResolver0 = new SimpleAbstractTypeResolver();
      DeserializerFactory deserializerFactory0 = beanDeserializerFactory0.withAbstractTypeResolver(simpleAbstractTypeResolver0);
      assertNotSame(beanDeserializerFactory0, deserializerFactory0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      PropertyName propertyName0 = beanDeserializerFactory0._findParamName((AnnotatedParameter) null, annotationIntrospector0);
      assertNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, (Type) null, annotationMap0, 2);
      PropertyName propertyName0 = beanDeserializerFactory0._findParamName(annotatedParameter0, annotationIntrospector0);
      assertNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      PropertyName propertyName0 = beanDeserializerFactory0._findImplicitParamName((AnnotatedParameter) null, annotationIntrospector0);
      assertNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      PropertyName propertyName0 = beanDeserializerFactory0._findExplicitParamName((AnnotatedParameter) null, (AnnotationIntrospector) null);
      assertNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      boolean boolean0 = beanDeserializerFactory0._hasExplicitParamName((AnnotatedParameter) null, annotationIntrospector0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, (Type) null, annotationMap0, 2);
      boolean boolean0 = beanDeserializerFactory0._hasExplicitParamName(annotatedParameter0, annotationIntrospector0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, (Type) null, annotationMap0, 2);
      boolean boolean0 = beanDeserializerFactory0._hasExplicitParamName(annotatedParameter0, (AnnotationIntrospector) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      Class<AsArrayTypeDeserializer> class0 = AsArrayTypeDeserializer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      CollectionType collectionType0 = beanDeserializerFactory0._mapAbstractCollectionType(simpleType0, deserializationConfig0);
      assertNull(collectionType0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.createKeyDeserializer(defaultDeserializationContext_Impl0, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<ArrayNode> class0 = ArrayNode.class;
      // Undeclared exception!
      try { 
        beanDeserializerFactory0.constructEnumResolver(class0, (DeserializationConfig) null, (AnnotatedMethod) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<AsWrapperTypeDeserializer> class0 = AsWrapperTypeDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      MapType mapType0 = MapType.construct(class0, simpleType0, simpleType0);
      // Undeclared exception!
      try { 
        beanDeserializerFactory0._findJsonValueFor((DeserializationConfig) null, mapType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BasicDeserializerFactory", e);
      }
  }
}