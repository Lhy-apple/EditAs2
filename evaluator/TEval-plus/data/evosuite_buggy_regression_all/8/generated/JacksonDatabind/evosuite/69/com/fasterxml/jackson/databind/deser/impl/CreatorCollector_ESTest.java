/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:15:06 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.impl.CreatorCollector;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BasicClassIntrospector;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.util.Annotations;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import java.util.HashMap;
import java.util.LinkedHashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CreatorCollector_ESTest extends CreatorCollector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0);
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, deserializationConfig0);
      creatorCollector0.addLongCreator((AnnotatedWithParams) null, true);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, deserializationConfig0);
      creatorCollector0.addLongCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, deserializationConfig0);
      creatorCollector0.addIntCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<BasicBeanDescription> class0 = BasicBeanDescription.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) simpleType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, simpleMixInResolver0);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      // Undeclared exception!
      try { 
        creatorCollector0.addDelegatingCreator((AnnotatedWithParams) null, (CreatorProperty[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0);
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, deserializationConfig0);
      creatorCollector0.addStringCreator((AnnotatedWithParams) null);
      assertFalse(creatorCollector0.hasPropertyBasedCreator());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0);
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, deserializationConfig0);
      creatorCollector0.addDoubleCreator((AnnotatedWithParams) null, false);
      assertFalse(creatorCollector0.hasPropertyBasedCreator());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla((-10));
      boolean boolean0 = creatorCollector_Vanilla0.canCreateUsingDefault();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(7);
      boolean boolean0 = creatorCollector_Vanilla0.canInstantiate();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorCollector creatorCollector0 = new CreatorCollector((BeanDescription) null, deserializationConfig0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      ObjectIdGenerators.UUIDGenerator objectIdGenerators_UUIDGenerator0 = new ObjectIdGenerators.UUIDGenerator();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2624), objectIdGenerators_UUIDGenerator0, (PropertyMetadata) null);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[2];
      creatorPropertyArray0[0] = creatorProperty0;
      creatorPropertyArray0[1] = creatorProperty0;
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, creatorPropertyArray0);
      assertFalse(creatorCollector0.hasPropertyBasedCreator());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 5, (Object) null, propertyMetadata0);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[8];
      creatorPropertyArray0[0] = creatorProperty0;
      // Undeclared exception!
      try { 
        creatorCollector0.addPropertyCreator((AnnotatedWithParams) null, creatorPropertyArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, (ClassIntrospector.MixInResolver) null);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      PropertyName propertyName0 = new PropertyName("15[bZW+quLoY<g.X.O");
      Annotations annotations0 = basicBeanDescription0.getClassAnnotations();
      Integer integer0 = new Integer(108);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(false, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotations0, (AnnotatedParameter) null, 108, creatorCollector0, propertyMetadata0);
      CreatorProperty[] creatorPropertyArray0 = new CreatorProperty[3];
      creatorPropertyArray0[0] = creatorProperty0;
      creatorPropertyArray0[1] = creatorProperty0;
      // Undeclared exception!
      try { 
        creatorCollector0.addPropertyCreator((AnnotatedWithParams) annotatedConstructor0, creatorPropertyArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Duplicate creator property \"15[bZW+quLoY<g.X.O\" (index 0 vs 1)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, (ClassIntrospector.MixInResolver) null);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter(annotatedConstructor0, simpleType0, annotationMap0, 846);
      creatorCollector0.addIncompeteParameter(annotatedParameter0);
      creatorCollector0.addIncompeteParameter(annotatedParameter0);
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, deserializationConfig0);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      creatorCollector0.verifyNonDup(annotatedConstructor0, 6, false);
      boolean boolean0 = creatorCollector0.hasDelegatingCreator();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, simpleMixInResolver0);
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      SettableBeanProperty[] settableBeanPropertyArray0 = new SettableBeanProperty[0];
      creatorCollector0.addPropertyCreator((AnnotatedWithParams) annotatedConstructor0, false, settableBeanPropertyArray0);
      boolean boolean0 = creatorCollector0.hasPropertyBasedCreator();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, (ClassIntrospector.MixInResolver) null);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      creatorCollector0.verifyNonDup(annotatedConstructor0, 8, true);
      ValueInstantiator valueInstantiator0 = creatorCollector0.constructValueInstantiator(deserializationConfig0);
      assertEquals("[simple type, class java.lang.String]", valueInstantiator0.getValueTypeDesc());
      assertFalse(valueInstantiator0.canInstantiate());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, (ClassIntrospector.MixInResolver) null);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      creatorCollector0.verifyNonDup(annotatedConstructor0, 8, true);
      // Undeclared exception!
      try { 
        creatorCollector0.verifyNonDup(annotatedConstructor0, 8, true);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 8
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, (ClassIntrospector.MixInResolver) null);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      creatorCollector0.addBooleanCreator((AnnotatedWithParams) annotatedConstructor0);
      // Undeclared exception!
      try { 
        creatorCollector0.addDoubleCreator((AnnotatedWithParams) annotatedConstructor0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Conflicting boolean creators: already had explicitly marked [constructor for java.lang.String, annotations: [null]], encountered [constructor for java.lang.String, annotations: [null]]
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, deserializationConfig0);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      creatorCollector0.addBooleanCreator((AnnotatedWithParams) annotatedConstructor0, true);
      creatorCollector0.addBooleanCreator((AnnotatedWithParams) annotatedConstructor0);
      assertFalse(creatorCollector0.hasDelegatingCreator());
      assertFalse(creatorCollector0.hasDefaultCreator());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BasicClassIntrospector basicClassIntrospector0 = new BasicClassIntrospector();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BasicBeanDescription basicBeanDescription0 = basicClassIntrospector0.forClassAnnotations(deserializationConfig0, simpleType0, simpleMixInResolver0);
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      CreatorCollector creatorCollector0 = new CreatorCollector(basicBeanDescription0, deserializationConfig0);
      creatorCollector0.verifyNonDup(annotatedConstructor0, 8, false);
      creatorCollector0.verifyNonDup(annotatedConstructor0, 8, true);
      assertFalse(creatorCollector0.hasPropertyBasedCreator());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(1);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.ArrayList", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.HashMap", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.util.LinkedHashMap", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(7);
      String string0 = creatorCollector_Vanilla0.getValueTypeDesc();
      assertEquals("java.lang.Object", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(1);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(7);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Unknown type 7
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.CreatorCollector$Vanilla", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(2);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      LinkedHashMap linkedHashMap0 = (LinkedHashMap)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertTrue(linkedHashMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CreatorCollector.Vanilla creatorCollector_Vanilla0 = new CreatorCollector.Vanilla(3);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      HashMap hashMap0 = (HashMap)creatorCollector_Vanilla0.createUsingDefault(defaultDeserializationContext_Impl0);
      assertTrue(hashMap0.isEmpty());
  }
}