/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:09:27 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.type.WritableTypeId;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationCollector;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.TypeResolutionContext;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.annotation.Annotation;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CreatorProperty_ESTest extends CreatorProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, (-459));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, (-459), asArrayTypeDeserializer0, propertyMetadata0);
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      try { 
        creatorProperty0.setAndReturn(byteArrayInputStream0, annotationMap0);
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
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, 0, annotationMap0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withName((PropertyName) null);
      assertEquals(0, settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      JsonDeserializer<InputStream> jsonDeserializer0 = (JsonDeserializer<InputStream>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      doReturn((Object) null).when(jsonDeserializer0).getNullValue(any(com.fasterxml.jackson.databind.DeserializationContext.class));
      CreatorProperty creatorProperty1 = new CreatorProperty(creatorProperty0, jsonDeserializer0, jsonDeserializer0);
      creatorProperty1.setFallbackSetter(creatorProperty1);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      char[] charArray0 = new char[9];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        creatorProperty1.deserializeAndSet(jsonParser0, deserializationContext0, annotationMap0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, (-459));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, (-459), asArrayTypeDeserializer0, propertyMetadata0);
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withNullProvider((NullValueProvider) null);
      assertEquals((-459), settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, 0, asArrayTypeDeserializer0, propertyMetadata0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonFactory jsonFactory0 = new JsonFactory();
      char[] charArray0 = new char[0];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0, (-4941), (-4941));
      try { 
        creatorProperty0.deserializeSetAndReturn(jsonParser0, defaultDeserializationContext_Impl0, annotationMap0);
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
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, (-459));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, (-459), asArrayTypeDeserializer0, propertyMetadata0);
      // Undeclared exception!
      try { 
        creatorProperty0.getDeclaringClass();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.AnnotatedParameter", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, (-459));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, (-459), asArrayTypeDeserializer0, propertyMetadata0);
      creatorProperty0.markAsIgnorable();
      assertTrue(creatorProperty0.isIgnorable());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      Object object1 = creatorProperty0.getInjectableValueId();
      assertEquals((-45), creatorProperty0.getCreatorIndex());
      assertNotNull(object1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, 0, asArrayTypeDeserializer0, propertyMetadata0);
      creatorProperty0.toString();
      assertEquals(0, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("k0)q");
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, javaType0, propertyName0, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, 0, propertyMetadata0, propertyMetadata0);
      // Undeclared exception!
      try { 
        creatorProperty0.inject((DeserializationContext) null, annotationMap0);
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
  public void test11()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, true, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, 6);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, 6, (Object) null, propertyMetadata0);
      creatorProperty0.isIgnorable();
      assertEquals(6, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, 0, annotationMap0, propertyMetadata0);
      JsonDeserializer<SimpleModule> jsonDeserializer0 = (JsonDeserializer<SimpleModule>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withValueDeserializer(jsonDeserializer0);
      assertEquals(0, settableBeanProperty0.getCreatorIndex());
      assertTrue(settableBeanProperty0.hasValueDeserializer());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, (-459));
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, (-459), asArrayTypeDeserializer0, propertyMetadata0);
      JsonDeserializer<Object> jsonDeserializer0 = SettableBeanProperty.MISSING_VALUE_DESERIALIZER;
      SettableBeanProperty settableBeanProperty0 = creatorProperty0.withValueDeserializer(jsonDeserializer0);
      assertSame(settableBeanProperty0, creatorProperty0);
      assertEquals((-459), settableBeanProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      creatorProperty0.fixAccess((DeserializationConfig) null);
      assertEquals((-45), creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-45), object0, (PropertyMetadata) null);
      creatorProperty0._fallbackSetter = (SettableBeanProperty) creatorProperty0;
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
  public void test16()  throws Throwable  {
      Class<InputStream> class0 = InputStream.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(resolvedRecursiveType0, (TypeFactory) null);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(resolvedRecursiveType0, classNameIdResolver0, (String) null, true, resolvedRecursiveType0);
      Class<Object> class1 = Object.class;
      Class<Integer> class2 = Integer.class;
      AnnotationCollector.TwoAnnotations annotationCollector_TwoAnnotations0 = new AnnotationCollector.TwoAnnotations(class1, (Annotation) null, class2, (Annotation) null);
      TypeResolutionContext.Basic typeResolutionContext_Basic0 = new TypeResolutionContext.Basic((TypeFactory) null, typeBindings0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, resolvedRecursiveType0, typeResolutionContext_Basic0, annotationMap0, 18);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, resolvedRecursiveType0, (PropertyName) null, asWrapperTypeDeserializer0, annotationCollector_TwoAnnotations0, annotatedParameter0, 18, (Object) null, propertyMetadata0);
      WritableTypeId.Inclusion writableTypeId_Inclusion0 = WritableTypeId.Inclusion.WRAPPER_OBJECT;
      // Undeclared exception!
      try { 
        creatorProperty0.inject((DeserializationContext) null, writableTypeId_Inclusion0);
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
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(javaType0, classNameIdResolver0, (String) null, false, javaType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, (TypeResolutionContext) null, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      CreatorProperty creatorProperty0 = new CreatorProperty((PropertyName) null, javaType0, (PropertyName) null, asArrayTypeDeserializer0, annotationMap0, annotatedParameter0, 0, annotationMap0, propertyMetadata0);
      Class<Annotation> class0 = Annotation.class;
      creatorProperty0.getAnnotation(class0);
      assertEquals(0, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("implicitly discovered", "'null', 'true' or 'false'");
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Object object0 = new Object();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, resolvedRecursiveType0, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-3758), object0, (PropertyMetadata) null);
      Class<Annotation> class1 = Annotation.class;
      creatorProperty0.getAnnotation(class1);
      assertEquals((-3758), creatorProperty0.getCreatorIndex());
  }
}
