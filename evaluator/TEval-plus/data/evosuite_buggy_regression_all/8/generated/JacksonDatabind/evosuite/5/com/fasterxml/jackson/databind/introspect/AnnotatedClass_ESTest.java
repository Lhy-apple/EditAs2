/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:59:47 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedField;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BasicClassIntrospector;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.util.Annotations;
import java.lang.annotation.Annotation;
import java.lang.reflect.Array;
import java.time.chrono.ChronoLocalDate;
import java.util.HashMap;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AnnotatedClass_ESTest extends AnnotatedClass_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Class<?> class0 = annotatedClass0.getRawType();
      assertEquals("boolean", class0.toString());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      int int0 = annotatedClass0.getModifiers();
      assertEquals(1041, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<String> class0 = String.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      // Undeclared exception!
      try { 
        annotatedClass0.getType((TypeBindings) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.Annotated", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Class<?> class1 = annotatedClass0.getAnnotated();
      assertFalse(class1.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedClass annotatedClass1 = annotatedClass0.withAnnotations(annotationMap0);
      assertEquals(1041, annotatedClass1.getModifiers());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      String string0 = annotatedClass0.getName();
      assertEquals("boolean", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      String string0 = annotatedClass0.toString();
      assertEquals("[AnnotedClass int]", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Iterable<Annotation> iterable0 = annotatedClass0.annotations();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<AnnotatedField> class0 = AnnotatedField.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Iterable<Annotation> iterable0 = annotatedClass0.annotations();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      annotatedClass0.getAllAnnotations();
      int int0 = annotatedClass0.getFieldCount();
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      basicBeanDescription0.getClassAnnotations();
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      beanSerializerBuilder0.getClassInfo();
      Class<String> class0 = String.class;
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<Integer> class0 = Integer.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Annotations annotations0 = annotatedClass0.getAnnotations();
      assertEquals(0, annotations0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      boolean boolean0 = annotatedClass0.hasAnnotations();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      boolean boolean0 = annotatedClass0.hasAnnotations();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<AnnotatedField> class0 = AnnotatedField.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      AnnotatedConstructor annotatedConstructor0 = annotatedClass0.getDefaultConstructor();
      assertNull(annotatedConstructor0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      int int0 = annotatedClass0.getMemberMethodCount();
      assertEquals(51, int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      int int0 = annotatedClass0.getMemberMethodCount();
      assertEquals(30, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      List<AnnotatedMethod> list0 = annotatedClass0.getStaticMethods();
      assertEquals(20, list0.size());
      
      int int0 = annotatedClass0.getFieldCount();
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SimpleModule simpleModule0 = new SimpleModule();
      Class<LongNode> class0 = LongNode.class;
      SimpleModule simpleModule1 = simpleModule0.setMixInAnnotation(class0, class0);
      ObjectMapper objectMapper1 = objectMapper0.registerModule(simpleModule1);
      objectMapper1.reader(class0);
      assertEquals("SimpleModule-157", simpleModule1.getModuleName());
      assertEquals("SimpleModule-157", simpleModule0.getModuleName());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SimpleModule simpleModule0 = new SimpleModule();
      SimpleModule simpleModule1 = simpleModule0.setMixInAnnotation(class0, class0);
      ObjectMapper objectMapper1 = objectMapper0.registerModule(simpleModule1);
      ObjectReader objectReader0 = objectMapper1.reader(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SimpleModule simpleModule0 = new SimpleModule();
      Class<TypeIdResolver> class0 = TypeIdResolver.class;
      Class<BasicBeanDescription> class1 = BasicBeanDescription.class;
      simpleModule0.setMixInAnnotation(class0, class1);
      objectMapper0.registerModule(simpleModule0);
      ObjectReader objectReader0 = objectMapper0.reader(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Class<JsonFactory.Feature>[] classArray0 = (Class<JsonFactory.Feature>[]) Array.newInstance(Class.class, 0);
      basicBeanDescription0.findMethod("", classArray0);
      Class<String> class0 = String.class;
      annotatedClass0._addConstructorMixIns(class0);
      assertEquals(17, annotatedClass0.getModifiers());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct((Class<?>) null, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Iterable<AnnotatedMethod> iterable0 = annotatedClass0.memberMethods();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<MinimalClassNameIdResolver> class0 = MinimalClassNameIdResolver.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Class<AnnotatedField> class1 = AnnotatedField.class;
      HashMap<String, AnnotatedField> hashMap0 = new HashMap<String, AnnotatedField>();
      annotatedClass0._addFieldMixIns(class1, class0, hashMap0);
      assertEquals(1, annotatedClass0.getModifiers());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<JsonFactory.Feature> class0 = JsonFactory.Feature.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      int int0 = annotatedClass0.getFieldCount();
      assertEquals(3, int0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<AnnotatedField> class0 = AnnotatedField.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      AnnotationMap annotationMap0 = annotatedClass0._collectRelevantAnnotations((Annotation[]) null);
      assertEquals(0, annotationMap0.size());
  }
}