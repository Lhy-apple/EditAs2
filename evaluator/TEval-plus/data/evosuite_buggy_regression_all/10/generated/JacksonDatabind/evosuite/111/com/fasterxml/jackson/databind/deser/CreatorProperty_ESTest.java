/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:43:01 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerBase;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.ext.NioPathDeserializer;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.TypeResolutionContext;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.annotation.Annotation;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CreatorProperty_ESTest extends CreatorProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertEquals((-45), settableBeanProperty0.getCreatorIndex());
      assertNotSame(settableBeanProperty0, creatorProperty0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("%ADkV>@|;h#:Ey2");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-28), (Object) null, (PropertyMetadata) null);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withName(propertyName0);
      assertEquals((-28), settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyName propertyName0 = BeanDeserializerBase.TEMP_PROPERTY_NAME;
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      try { 
        creatorProperty0.set(class0, object0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property '#temporary-name'
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      NioPathDeserializer nioPathDeserializer0 = new NioPathDeserializer();
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withNullProvider(nioPathDeserializer0);
      assertEquals((-45), settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        creatorProperty0.deserializeSetAndReturn((JsonParser) null, defaultDeserializationContext_Impl0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property 'implicitly discovered'
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 0, (Object) null, (PropertyMetadata) null);
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
      PropertyName propertyName0 = new PropertyName("%ADkV>@|;h#:Ey2");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-16), (Object) null, (PropertyMetadata) null);
      creatorProperty0.markAsIgnorable();
      assertTrue(creatorProperty0.isIgnorable());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("k0)q");
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, 285, annotationMap0, propertyMetadata0);
      Object object0 = creatorProperty0.getInjectableValueId();
      assertNotNull(object0);
      assertEquals(285, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("%ADkV>@|;h#:Ey2");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-16), (Object) null, (PropertyMetadata) null);
      String string0 = creatorProperty0.toString();
      assertEquals((-16), creatorProperty0.getCreatorIndex());
      assertEquals("[creator property, name '%ADkV>@|;h#:Ey2'; inject id 'null']", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true'or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      // Undeclared exception!
      try { 
        creatorProperty0.inject((DeserializationContext) null, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.CreatorProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      creatorProperty0.setFallbackSetter(creatorProperty0);
      // Undeclared exception!
      try { 
        creatorProperty0.fixAccess((DeserializationConfig) null);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory(objectMapper0);
      char[] charArray0 = new char[0];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0, (-967), 69);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        creatorProperty0.deserializeAndSet(jsonParser0, defaultDeserializationContext_Impl0, propertyName0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No fallback setter/field defined for creator property 'implicitly discovered'
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      int int0 = creatorProperty0.getCreatorIndex();
      assertEquals((-45), int0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      creatorProperty0.isIgnorable();
      assertEquals((-45), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Iterable<Annotation> iterable0 = annotationMap0.annotations();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), iterable0, (PropertyMetadata) null);
      JsonDeserializer<Object> jsonDeserializer0 = creatorProperty0._valueDeserializer;
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withValueDeserializer(jsonDeserializer0);
      assertEquals((-45), settableBeanProperty0.getCreatorIndex());
      assertSame(settableBeanProperty0, creatorProperty0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      creatorProperty0.fixAccess((DeserializationConfig) null);
      assertEquals((-45), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "implicitly discovered");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), (Object) null, (PropertyMetadata) null);
      // Undeclared exception!
      try { 
        creatorProperty0.inject((DeserializationContext) null, resolvedRecursiveType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.CreatorProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("k0)q");
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, 285, annotationMap0, propertyMetadata0);
      Class<Annotation> class0 = Annotation.class;
      creatorProperty0.getAnnotation(class0);
      assertEquals(285, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "implicitly discovered");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      Class<Annotation> class1 = Annotation.class;
      creatorProperty0.getAnnotation(class1);
      assertEquals((-45), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      creatorProperty0.setFallbackSetter(creatorProperty0);
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      // Undeclared exception!
      creatorProperty0.setAndReturn(byteArrayInputStream0, byteArrayInputStream0);
  }
}