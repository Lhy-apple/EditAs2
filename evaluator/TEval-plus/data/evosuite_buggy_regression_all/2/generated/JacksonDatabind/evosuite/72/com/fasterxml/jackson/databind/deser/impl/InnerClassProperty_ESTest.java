/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:03:29 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.impl.InnerClassProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.io.IOException;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import java.util.Enumeration;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class InnerClassProperty_ESTest extends InnerClassProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(simpleType0, simpleType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, (-414));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionLikeType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, (-414), class0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      char[] charArray0 = new char[18];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0, (-414), (-414));
      // Undeclared exception!
      try { 
        innerClassProperty0.deserializeAndSet(jsonParser0, defaultDeserializationContext_Impl0, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, 11);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 11, propertyMetadata0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        innerClassProperty0.deserializeSetAndReturn((JsonParser) null, defaultDeserializationContext_Impl0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.SettableBeanProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("F/1ys$3c*m");
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(simpleType0, simpleType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, (-404));
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionLikeType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, (-404), sequenceInputStream0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      int int0 = innerClassProperty0.getPropertyIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, (-414));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, (-414), simpleType0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.set(propertyMetadata0, annotationMap0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No fallback setter/field defined: can not use creator property for com.fasterxml.jackson.databind.deser.CreatorProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.CreatorProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, 416);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 416, annotatedParameter0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      InnerClassProperty innerClassProperty1 = innerClassProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertFalse(innerClassProperty1.hasValueDeserializer());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, (-397));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, (-397), (Object) null, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      innerClassProperty0.assignIndex((-397));
      assertEquals((-397), innerClassProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, 568);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 568, annotatedParameter0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      Class<Annotation> class1 = Annotation.class;
      Annotation annotation0 = innerClassProperty0.getAnnotation(class1);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, (-404));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, (-404), propertyName0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.setAndReturn(propertyMetadata0, class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No fallback setter/field defined: can not use creator property for com.fasterxml.jackson.databind.deser.CreatorProperty
         //
         verifyException("com.fasterxml.jackson.databind.deser.CreatorProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, 416);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, 416, annotatedParameter0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.readResolve();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Missing constructor (broken JDK (de)serialization?)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, (-404));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, (-404), propertyName0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      InnerClassProperty innerClassProperty1 = innerClassProperty0.withName(propertyName0);
      assertFalse(innerClassProperty1.isRequired());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("F/1ys$3c*m");
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(simpleType0, simpleType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, collectionLikeType0, annotationMap0, (-404));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionLikeType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, (-404), collectionLikeType0, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      AnnotatedParameter annotatedParameter1 = (AnnotatedParameter)innerClassProperty0.getMember();
      assertEquals((-404), annotatedParameter1.getIndex());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, 2);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, (TypeIdResolver) null, "", false, class0);
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, asWrapperTypeDeserializer0, annotationMap0, annotatedParameter0, 2522, (Object) null, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser((char[]) null, 1, 1);
      try { 
        innerClassProperty0.deserializeAndSet(jsonParser0, defaultDeserializationContext_Impl0, jsonParser0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unexpected token (null), expected START_OBJECT: need JSON Object to contain As.WRAPPER_OBJECT type information for class java.lang.Object
         //  at [Source: UNKNOWN; line: 1, column: 0]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, (-397));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, (TypeDeserializer) null, annotationMap0, annotatedParameter0, (-397), (Object) null, propertyMetadata0);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.writeReplace();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null constructor not allowed
         //
         verifyException("com.fasterxml.jackson.databind.introspect.AnnotatedConstructor", e);
      }
  }
}