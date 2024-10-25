/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:06:43 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.type.TypeFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeDeserializerBase_ESTest extends TypeDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(javaType0, (TypeIdResolver) null, "91%VvVBR%", false, javaType0);
      String string0 = asWrapperTypeDeserializer0.getPropertyName();
      assertEquals("91%VvVBR%", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(javaType0, (TypeIdResolver) null, "+*[`L<mlQE%FoYvj>", false, javaType0);
      Class<?> class0 = asWrapperTypeDeserializer0.getDefaultImpl();
      assertEquals("+*[`L<mlQE%FoYvj>", asWrapperTypeDeserializer0.getPropertyName());
      assertNotNull(class0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, " $~", true, javaType0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._handleMissingTypeId((DeserializationContext) null, " $~");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "", true, (JavaType) null);
      TypeIdResolver typeIdResolver0 = asPropertyTypeDeserializer0.getTypeIdResolver();
      assertNull(typeIdResolver0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, (String) null, false, javaType0);
      String string0 = asPropertyTypeDeserializer0.toString();
      assertEquals("[com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer; base-type:[simple type, class java.lang.Object]; id-resolver: null]", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(javaType0, (TypeIdResolver) null, "", true, javaType0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(javaType0, (TypeIdResolver) null, "", true, javaType0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0.deserializeTypedFromObject(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(javaType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(javaType0, minimalClassNameIdResolver0, "", false, javaType0);
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      AsWrapperTypeDeserializer asWrapperTypeDeserializer1 = new AsWrapperTypeDeserializer(asWrapperTypeDeserializer0, beanProperty_Bogus0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer1._handleUnknownTypeId((DeserializationContext) null, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(javaType0, (TypeIdResolver) null, "91%VvVBR%", false, javaType0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, (DeserializationContext) null, "91%VvVBR%");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "=CI", false, javaType0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._findDefaultImplDeserializer((DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, (TypeIdResolver) null, "", false, (JavaType) null);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.NullifyingDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(javaType0, (TypeIdResolver) null, "91%VvVBR%", false, javaType0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, (DeserializationContext) null, javaType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(javaType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(javaType0, minimalClassNameIdResolver0, "", false, javaType0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._handleUnknownTypeId((DeserializationContext) null, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }
}
