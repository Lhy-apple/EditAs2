/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:46:49 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeDeserializerBase_ESTest extends TypeDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.WRAPPER_OBJECT;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(arrayType0, (TypeIdResolver) null, "[SJNDIj+L3[o", true, arrayType0, jsonTypeInfo_As0);
      String string0 = asPropertyTypeDeserializer0.getPropertyName();
      assertEquals("[SJNDIj+L3[o", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(mapType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(mapType0, classNameIdResolver0, "+Infinity", true, mapType0);
      Class<?> class1 = asExternalTypeDeserializer0.getDefaultImpl();
      assertNotNull(class1);
      assertEquals("+Infinity", asExternalTypeDeserializer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer((JavaType) null, (TypeIdResolver) null, "", true, (JavaType) null);
      // Undeclared exception!
      try { 
        asExternalTypeDeserializer0._handleMissingTypeId(defaultDeserializationContext_Impl0, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(resolvedRecursiveType0, typeFactory0);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(resolvedRecursiveType0, minimalClassNameIdResolver0, "Q", false, resolvedRecursiveType0);
      TypeIdResolver typeIdResolver0 = asArrayTypeDeserializer0.getTypeIdResolver();
      assertEquals("Q", asArrayTypeDeserializer0.getPropertyName());
      assertNotNull(typeIdResolver0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer((JavaType) null, (TypeIdResolver) null, "known type ids = ", true, (JavaType) null);
      String string0 = asExternalTypeDeserializer0.toString();
      assertEquals("[com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer; base-type:null; id-resolver: null]", string0);
      assertEquals("known type ids = ", asExternalTypeDeserializer0.getPropertyName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer((JavaType) null, (TypeIdResolver) null, "J{1H", false, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      // Undeclared exception!
      try { 
        asExternalTypeDeserializer0.deserializeTypedFromAny(jsonParser0, defaultDeserializationContext_Impl0);
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
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, (TypeIdResolver) null, "type ids are not statically known", false, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, (TypeIdResolver) null, "type ids are not statically known", false, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0, "type ids are not statically known");
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
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(mapType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(mapType0, minimalClassNameIdResolver0, "T;9X@Kgti,de)b+hS", true, mapType0);
      // Undeclared exception!
      try { 
        asExternalTypeDeserializer0._findDefaultImplDeserializer((DeserializationContext) null);
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
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, (TypeIdResolver) null, "type ids are not statically known", false, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
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
  public void test10()  throws Throwable  {
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer((JavaType) null, (TypeIdResolver) null, "type ids are not statically known", false, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(javaType0, classNameIdResolver0, "D!&lMGvBG.@QxW.0gt4", false, javaType0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        asExternalTypeDeserializer0._handleUnknownTypeId(defaultDeserializationContext_Impl0, "D!&lMGvBG.@QxW.0gt4");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, typeFactory0);
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(javaType0, classNameIdResolver0, "D!&lMGvBG.@QxW.0gt4", false, javaType0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer(asExternalTypeDeserializer0, beanProperty_Bogus0);
      // Undeclared exception!
      try { 
        asArrayTypeDeserializer0._handleUnknownTypeId(defaultDeserializationContext_Impl0, "D!&lMGvBG.@QxW.0gt4");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }
}