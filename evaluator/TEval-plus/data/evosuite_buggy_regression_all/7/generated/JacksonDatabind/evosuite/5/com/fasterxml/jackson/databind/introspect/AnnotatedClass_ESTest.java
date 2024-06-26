/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:51:49 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedField;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethodMap;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BasicClassIntrospector;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.util.Annotations;
import java.lang.annotation.Annotation;
import java.lang.reflect.Array;
import java.time.chrono.ChronoLocalDate;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AnnotatedClass_ESTest extends AnnotatedClass_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Class<?> class0 = annotatedClass0.getRawType();
      assertFalse(class0.isEnum());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<AnnotatedMethod> class0 = AnnotatedMethod.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      ObjectReader objectReader0 = objectMapper0.reader((JavaType) simpleType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      boolean boolean0 = annotatedClass0.isPublic();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<AnnotatedMethod> class0 = AnnotatedMethod.class;
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
  public void test04()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      annotatedClass0.getName();
      int int0 = annotatedClass0.getMemberMethodCount();
      assertEquals(51, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<AnnotatedConstructor> class0 = AnnotatedConstructor.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      AnnotatedMethodMap annotatedMethodMap0 = new AnnotatedMethodMap();
      Class<ObjectNode> class1 = ObjectNode.class;
      annotatedClass0._addMemberMethods(class1, annotatedMethodMap0, class0, annotatedMethodMap0);
      assertEquals(17, annotatedClass0.getModifiers());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedClass annotatedClass1 = annotatedClass0.withAnnotations(annotationMap0);
      assertNotSame(annotatedClass1, annotatedClass0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<AnnotatedField> class0 = AnnotatedField.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, (AnnotationIntrospector) null, (ClassIntrospector.MixInResolver) null);
      Class<?> class1 = annotatedClass0.getAnnotated();
      assertFalse(class1.isSynthetic());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      String string0 = annotatedClass0.toString();
      assertEquals("[AnnotedClass java.lang.String]", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      annotatedClass0.annotations();
      int int0 = annotatedClass0.getFieldCount();
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      AnnotationMap annotationMap0 = annotatedClass0.getAllAnnotations();
      assertEquals(0, annotationMap0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      boolean boolean0 = annotatedClass0.hasAnnotations();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      Object object0 = basicBeanDescription0.instantiateBean(false);
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      List<AnnotatedMethod> list0 = annotatedClass0.getStaticMethods();
      assertNotNull(list0);
      
      boolean boolean0 = annotatedClass0.hasAnnotations();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      int int0 = annotatedClass0.getMemberMethodCount();
      assertEquals(51, int0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      Class<String>[] classArray0 = (Class<String>[]) Array.newInstance(Class.class, 0);
      // Undeclared exception!
      try { 
        basicBeanDescription0.findMethod((String) null, classArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.MemberKey", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      Class<AnnotatedConstructor>[] classArray0 = (Class<AnnotatedConstructor>[]) Array.newInstance(Class.class, 2);
      basicBeanDescription0.findMethod("Hf!#N>t[YC98fS$Y", classArray0);
      Annotations annotations0 = basicBeanDescription0.getClassAnnotations();
      assertEquals(0, annotations0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      int int0 = annotatedClass0.getFieldCount();
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<String> class0 = String.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      List<AnnotatedConstructor> list0 = annotatedClass0.getConstructors();
      assertNotNull(list0);
      assertEquals(15, list0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      int int0 = annotatedClass0.getMemberMethodCount();
      assertEquals(30, int0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Iterable<AnnotatedField> iterable0 = annotatedClass0.fields();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<AnnotatedField> class0 = AnnotatedField.class;
      Class<AnnotatedMethod> class1 = AnnotatedMethod.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class1, (AnnotationIntrospector) null, (ClassIntrospector.MixInResolver) null);
      AnnotationMap annotationMap0 = new AnnotationMap();
      annotatedClass0._addClassMixIns(annotationMap0, class0, class1);
      assertEquals(0, annotationMap0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Class<String> class0 = String.class;
      annotatedClass0._addConstructorMixIns(class0);
      assertEquals(17, annotatedClass0.getModifiers());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct((Class<?>) null, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      int int0 = annotatedClass0.getMemberMethodCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      AnnotatedMethodMap annotatedMethodMap0 = new AnnotatedMethodMap();
      Class<String> class0 = String.class;
      Class<Annotation> class1 = Annotation.class;
      annotatedClass0._addMemberMethods(class0, annotatedMethodMap0, class1, annotatedMethodMap0);
      assertEquals(17, annotatedClass0.getModifiers());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<ObjectNode> class0 = ObjectNode.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      HashMap<String, AnnotatedField> hashMap0 = new HashMap<String, AnnotatedField>();
      annotatedClass0._addFieldMixIns(class0, class0, hashMap0);
      assertTrue(hashMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Class<Annotation> class0 = Annotation.class;
      LinkedHashMap<String, AnnotatedField> linkedHashMap0 = new LinkedHashMap<String, AnnotatedField>();
      Class<AnnotatedMethod> class1 = AnnotatedMethod.class;
      annotatedClass0._findFields(class1, linkedHashMap0);
      annotatedClass0._addFieldMixIns(class0, class1, linkedHashMap0);
      assertEquals(3, linkedHashMap0.size());
      assertFalse(linkedHashMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<JsonTypeInfo.As> class0 = JsonTypeInfo.As.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Iterable<AnnotatedField> iterable0 = annotatedClass0.fields();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<SimpleObjectIdResolver> class0 = SimpleObjectIdResolver.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      AnnotationMap annotationMap0 = annotatedClass0._collectRelevantAnnotations((Annotation[]) null);
      assertEquals(0, annotationMap0.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<ObjectNode> class0 = ObjectNode.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      int int0 = annotatedClass0.getMemberMethodCount();
      assertEquals(139, int0);
  }
}
