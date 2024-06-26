/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:43:32 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.impl.InnerClassProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class InnerClassProperty_ESTest extends InnerClassProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationMap annotationMap0 = new AnnotationMap();
      Class<InnerClassProperty> class0 = InnerClassProperty.class;
      LinkedList<JavaType> linkedList0 = new LinkedList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) linkedList0);
      JavaType[] javaTypeArray0 = new JavaType[2];
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class0, typeBindings0, (JavaType) null, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionLikeType0, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(collectionLikeType0, classNameIdResolver0, "ALLOW_FINAL_FIELDS_AS_MUTATORS", false, class0);
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, collectionLikeType0, propertyName0, asPropertyTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, 0, "ALLOW_FINAL_FIELDS_AS_MUTATORS", (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonParser jsonParser0 = jsonFactory0.createParser("");
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        innerClassProperty0.deserializeAndSet(jsonParser0, deserializationContext0, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 65536, (Object) null, (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        innerClassProperty0.deserializeSetAndReturn((JsonParser) null, defaultDeserializationContext_Impl0, (Object) null);
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
      PropertyName propertyName0 = new PropertyName("v?hX@6I,1aXOGW-");
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 668, annotationMap0, (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      int int0 = innerClassProperty0.getPropertyIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-1905), propertyName0, (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.set((Object) null, (Object) null);
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
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 3205, (Object) null, (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      InnerClassProperty innerClassProperty1 = innerClassProperty0.withValueDeserializer((JsonDeserializer<?>) null);
      assertFalse(innerClassProperty1.isVirtual());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 1147, annotationMap0, (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      innerClassProperty0.assignIndex(1147);
      assertEquals(1147, innerClassProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 1147, annotationMap0, (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      Class<Annotation> class0 = Annotation.class;
      Annotation annotation0 = innerClassProperty0.getAnnotation(class0);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-1475), propertyName0, (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      // Undeclared exception!
      try { 
        innerClassProperty0.setAndReturn(propertyName0, propertyName0);
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
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 1, propertyName0, (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      InnerClassProperty innerClassProperty1 = innerClassProperty0.withName(propertyName0);
      assertNotSame(innerClassProperty1, innerClassProperty0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, (-2547), annotationMap0, (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      AnnotatedMember annotatedMember0 = innerClassProperty0.getMember();
      assertNull(annotatedMember0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 665, annotationMap0, (PropertyMetadata) null);
      InnerClassProperty innerClassProperty0 = new InnerClassProperty(creatorProperty0, (Constructor<?>) null);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        innerClassProperty0.deserializeAndSet(jsonParser0, deserializationContext0, annotationMap0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.InnerClassProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationMap annotationMap0 = new AnnotationMap();
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, (JavaType) null, propertyName0, (TypeDeserializer) null, annotationMap0, (AnnotatedParameter) null, 3205, (Object) null, (PropertyMetadata) null);
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
