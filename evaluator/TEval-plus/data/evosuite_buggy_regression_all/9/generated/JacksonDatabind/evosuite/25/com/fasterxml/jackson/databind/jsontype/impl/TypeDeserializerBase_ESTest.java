/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:43:26 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeDeserializerBase_ESTest extends TypeDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Map> class0 = Map.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(mapType0, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(mapType0, minimalClassNameIdResolver0, "; id-resolver: ", false, class0);
      String string0 = asPropertyTypeDeserializer0.getPropertyName();
      assertEquals("; id-resolver: ", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<CoreXMLDeserializers.Std> class0 = CoreXMLDeserializers.Std.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "E9t>R", true, class0);
      TypeIdResolver typeIdResolver0 = asPropertyTypeDeserializer0.getTypeIdResolver();
      assertNull(typeIdResolver0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "E9t>R", true, class0);
      String string0 = asPropertyTypeDeserializer0.toString();
      assertEquals("[com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer; base-type:[simple type, class com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer]; id-resolver: null]", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "`2", false, class0);
      String string0 = asPropertyTypeDeserializer0.baseTypeName();
      assertEquals("com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "E9t>R", true, class0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ArrayNode arrayNode0 = objectNode0.withArray("");
      JsonParser jsonParser0 = arrayNode0.traverse();
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._deserializeWithNativeTypeId(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper.DefaultTyping objectMapper_DefaultTyping0 = ObjectMapper.DefaultTyping.NON_FINAL;
      ObjectMapper objectMapper1 = objectMapper0.enableDefaultTyping(objectMapper_DefaultTyping0);
      ObjectReader objectReader0 = objectMapper1.readerFor((JavaType) simpleType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "`2", false, class0);
      Class<?> class1 = asPropertyTypeDeserializer0.getDefaultImpl();
      assertEquals("class com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer", class1.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<Object> class1 = Object.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "", false, class1);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, (DeserializationContext) null, "");
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
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, (String) null, false, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, defaultDeserializationContext_Impl0, simpleType0);
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
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, (TypeIdResolver) null, "com.fasterxml.jackson.databind.deser.std.StdDelegatingDeserializer", false, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._handleUnknownTypeId((DeserializationContext) null, "com.fasterxml.jackson.databind.deser.std.StdDelegatingDeserializer", (TypeIdResolver) null, simpleType0);
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
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, (TypeFactory) null);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, classNameIdResolver0, "^l9Bx", false, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._handleUnknownTypeId((DeserializationContext) null, "^l9Bx", classNameIdResolver0, simpleType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }
}