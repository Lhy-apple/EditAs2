/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:36:31 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedField;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethodMap;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BasicClassIntrospector;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.type.TypeBindings;
import java.lang.annotation.Annotation;
import java.lang.reflect.Array;
import java.time.chrono.ChronoLocalDate;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
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
      assertFalse(class0.isInterface());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
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
  public void test02()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedClass annotatedClass1 = annotatedClass0.withAnnotations(annotationMap0);
      assertNotSame(annotatedClass0, annotatedClass1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      String string0 = annotatedClass0.getName();
      assertEquals("boolean", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_DEFAULT;
      objectMapper0.setSerializationInclusion(jsonInclude_Include0);
      Class<AnnotatedConstructor> class0 = AnnotatedConstructor.class;
      objectMapper0.canSerialize(class0, atomicReference0);
      assertEquals("com.fasterxml.jackson.databind.JsonMappingException: Class com.fasterxml.jackson.databind.introspect.AnnotatedConstructor has no default constructor; can not instantiate default bean value to support 'properties=JsonSerialize.Inclusion.NON_DEFAULT' annotation", atomicReference0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      String string0 = annotatedClass0.toString();
      assertEquals("[AnnotedClass int]", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Iterable<Annotation> iterable0 = annotatedClass0.annotations();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<AnnotatedField> class0 = AnnotatedField.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Iterable<Annotation> iterable0 = annotatedClass0.annotations();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      AnnotationMap annotationMap0 = annotatedClass0.getAllAnnotations();
      assertEquals(0, annotationMap0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      annotatedClass0.getAnnotations();
      boolean boolean0 = annotatedClass0.isPublic();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      boolean boolean0 = annotatedClass0.hasAnnotations();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<AnnotatedField> class0 = AnnotatedField.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      AnnotatedConstructor annotatedConstructor0 = annotatedClass0.getDefaultConstructor();
      assertNull(annotatedConstructor0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      List<AnnotatedConstructor> list0 = annotatedClass0.getConstructors();
      assertNotNull(list0);
      assertEquals(15, list0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      // Undeclared exception!
      try { 
        basicBeanDescription0.getFactoryMethods();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.BasicBeanDescription", e);
      }
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
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      beanSerializerBuilder0.createDummy();
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      basicBeanDescription0.findMethod("7[t6vSLY", (Class<?>[]) null);
      Class<String> class0 = String.class;
      annotatedClass0._addFactoryMixIns(class0);
      assertEquals(17, annotatedClass0.getModifiers());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<MinimalClassNameIdResolver> class0 = MinimalClassNameIdResolver.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Class<String>[] classArray0 = (Class<String>[]) Array.newInstance(Class.class, 0);
      AnnotatedMethod annotatedMethod0 = annotatedClass0.findMethod("start an object", classArray0);
      assertNull(annotatedMethod0);
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      Class<ObjectIdGenerators.IntSequenceGenerator> class0 = ObjectIdGenerators.IntSequenceGenerator.class;
      Class<AnnotatedConstructor> class1 = AnnotatedConstructor.class;
      objectMapper0.addMixInAnnotations(class0, class1);
      boolean boolean0 = objectMapper0.canSerialize(class0, atomicReference0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      Class<JsonFactory.Feature> class0 = JsonFactory.Feature.class;
      objectMapper0.addMixInAnnotations(class0, class0);
      boolean boolean0 = objectMapper0.canSerialize(class0, atomicReference0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      Class<AnnotatedField> class0 = AnnotatedField.class;
      Class<ChronoLocalDate> class1 = ChronoLocalDate.class;
      objectMapper0.addMixInAnnotations(class1, class0);
      boolean boolean0 = objectMapper0.canSerialize(class1, atomicReference0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      Class<AnnotatedField> class0 = AnnotatedField.class;
      Class<Object> class1 = Object.class;
      objectMapper0.addMixInAnnotations(class1, class1);
      boolean boolean0 = objectMapper0.canSerialize(class0, atomicReference0);
      assertTrue(boolean0);
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      Class<AnnotatedField> class0 = AnnotatedField.class;
      Class<LongNode> class1 = LongNode.class;
      objectMapper0.addMixInAnnotations(class0, class1);
      boolean boolean0 = objectMapper0.canSerialize(class0, atomicReference0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct((Class<?>) null, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Iterable<AnnotatedMethod> iterable0 = annotatedClass0.memberMethods();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<Integer> class0 = Integer.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Class<AnnotatedParameter> class1 = AnnotatedParameter.class;
      AnnotatedMethodMap annotatedMethodMap0 = new AnnotatedMethodMap();
      Class<Annotation> class2 = Annotation.class;
      annotatedClass0._addMemberMethods(class1, annotatedMethodMap0, class2, annotatedMethodMap0);
      assertEquals(17, annotatedClass0.getModifiers());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      Class<JsonFactory.Feature> class0 = JsonFactory.Feature.class;
      boolean boolean0 = objectMapper0.canSerialize(class0, atomicReference0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<MinimalClassNameIdResolver> class0 = MinimalClassNameIdResolver.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      AnnotationMap annotationMap0 = annotatedClass0._collectRelevantAnnotations((Annotation[]) null);
      assertEquals(0, annotationMap0.size());
  }
}