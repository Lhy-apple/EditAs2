/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:07:24 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationCollector;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonObjectFormatVisitor;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.node.DoubleNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.sql.SQLNonTransientException;
import java.sql.SQLRecoverableException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SettableBeanProperty_ESTest extends SettableBeanProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      SQLNonTransientException sQLNonTransientException0 = new SQLNonTransientException();
      try { 
        creatorProperty0._throwAsIOE(sQLNonTransientException0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // N/A
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      int int0 = creatorProperty0.getPropertyIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      Class<Annotation> class2 = Annotation.class;
      creatorProperty0.getContextAnnotation(class2);
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      creatorProperty0.getObjectIdInfo();
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1137));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      try { 
        creatorProperty0.set(boolean0, (Object) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property ''
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
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
  public void test06()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1137));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, (-333), class0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withName(propertyName0);
      assertEquals((-1), settableBeanProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      creatorProperty0.setObjectIdInfo((ObjectIdInfo) null);
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      PropertyName propertyName1 = creatorProperty0.getFullName();
      assertSame(propertyName1, propertyName0);
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1135));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      creatorProperty0.getNullValueProvider();
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1137));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      SQLRecoverableException sQLRecoverableException0 = new SQLRecoverableException();
      try { 
        creatorProperty0._throwAsIOE(sQLRecoverableException0, (Object) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // N/A
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      creatorProperty0.getManagedReferenceName();
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ArrayType arrayType0 = ArrayType.construct((JavaType) simpleType0, (TypeBindings) null);
      Class<Integer> class1 = Integer.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class0, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, ">mqrP6D0Qi", integer0, "");
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(arrayType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, classNameIdResolver0, "com.fasterxml.jackson.annotation.JsonFormat$Feature", false, (JavaType) null);
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, asWrapperTypeDeserializer0, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, (-1137), propertyMetadata0, propertyMetadata0);
      boolean boolean1 = creatorProperty0.hasValueDeserializer();
      assertTrue(creatorProperty0.hasValueTypeDeserializer());
      assertFalse(boolean1);
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct((Boolean) null, "99?}DR}T)tW+|jy;zK", integer0, "");
      DoubleNode doubleNode0 = DoubleNode.valueOf(0.0);
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, (-1178), doubleNode0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertEquals((-1), settableBeanProperty0.getPropertyIndex());
      assertNotSame(settableBeanProperty0, creatorProperty0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonObjectFormatVisitor.Base jsonObjectFormatVisitor_Base0 = new JsonObjectFormatVisitor.Base(defaultSerializerProvider_Impl0);
      creatorProperty0.depositSchemaProperty(jsonObjectFormatVisitor_Base0, defaultSerializerProvider_Impl0);
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ArrayType arrayType0 = ArrayType.construct((JavaType) simpleType0, (TypeBindings) null);
      Class<Integer> class1 = Integer.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class0, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, ">mqrP6D0Qi", integer0, "");
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(arrayType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, classNameIdResolver0, "com.fasterxml.jackson.annotation.JsonFormat$Feature", false, (JavaType) null);
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, asWrapperTypeDeserializer0, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, (-1137), propertyMetadata0, propertyMetadata0);
      JsonDeserializer<DoubleNode> jsonDeserializer0 = (JsonDeserializer<DoubleNode>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      CreatorProperty creatorProperty1 = new CreatorProperty(creatorProperty0, jsonDeserializer0, jsonDeserializer0);
      boolean boolean1 = creatorProperty1.hasValueDeserializer();
      assertEquals((-1), creatorProperty1.getPropertyIndex());
      assertTrue(creatorProperty1.hasValueTypeDeserializer());
      assertTrue(boolean1);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      boolean boolean1 = creatorProperty0.hasValueTypeDeserializer();
      assertFalse(boolean1);
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ArrayType arrayType0 = ArrayType.construct((JavaType) simpleType0, (TypeBindings) null);
      Class<Integer> class1 = Integer.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class0, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, ">mqrP6D0Qi", integer0, "");
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(arrayType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, classNameIdResolver0, "com.fasterxml.jackson.annotation.JsonFormat$Feature", false, (JavaType) null);
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, asWrapperTypeDeserializer0, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, (-1137), propertyMetadata0, propertyMetadata0);
      boolean boolean1 = creatorProperty0.hasValueTypeDeserializer();
      assertEquals((-1), creatorProperty0.getPropertyIndex());
      assertTrue(boolean1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      JsonDeserializer<Object> jsonDeserializer0 = creatorProperty0.getValueDeserializer();
      assertNull(jsonDeserializer0);
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1135));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      boolean boolean1 = creatorProperty0.visibleInView(class0);
      assertEquals((-1), creatorProperty0.getPropertyIndex());
      assertTrue(boolean1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      boolean boolean1 = creatorProperty0.hasViews();
      assertEquals((-1), creatorProperty0.getPropertyIndex());
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-342));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn(integer0).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withValueDeserializer(jsonDeserializer0);
      assertTrue(settableBeanProperty0.hasValueDeserializer());
      
      JsonFactory jsonFactory0 = new JsonFactory();
      byte[] byteArray0 = new byte[6];
      JsonParser jsonParser0 = jsonFactory0.createParser(byteArray0, 499, 0);
      Object object0 = settableBeanProperty0.deserialize(jsonParser0, (DeserializationContext) null);
      assertNotNull(object0);
      assertEquals((-1), settableBeanProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1178));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, 499, class0, propertyMetadata0);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      doReturn((Object) null).when(jsonDeserializer0).getNullValue(any(com.fasterxml.jackson.databind.DeserializationContext.class));
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withValueDeserializer(jsonDeserializer0);
      JsonFactory jsonFactory0 = new JsonFactory();
      byte[] byteArray0 = new byte[6];
      JsonParser jsonParser0 = jsonFactory0.createParser(byteArray0, (-1178), 0);
      settableBeanProperty0.deserialize(jsonParser0, (DeserializationContext) null);
      assertEquals((-1), settableBeanProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<SimpleObjectIdResolver> class1 = SimpleObjectIdResolver.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class1, (Annotation) null);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(simpleType0);
      Boolean boolean0 = Boolean.valueOf("");
      Integer integer0 = new Integer((-1137));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(boolean0, "", integer0, "");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationCollector_TwoAnnotations0, (AnnotatedParameter) null, (-333), class0, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory();
      char[] charArray0 = new char[0];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0);
      // Undeclared exception!
      try { 
        creatorProperty0.deserializeWith(jsonParser0, (DeserializationContext) null, valueInstantiator_Base0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.JsonDeserializer", e);
      }
  }
}