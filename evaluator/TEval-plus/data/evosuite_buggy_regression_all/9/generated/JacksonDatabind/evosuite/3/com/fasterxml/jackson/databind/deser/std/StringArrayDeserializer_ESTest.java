/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:38:55 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.util.JsonParserSequence;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.lang.reflect.Type;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringArrayDeserializer_ESTest extends StringArrayDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      JsonFactory jsonFactory0 = new JsonFactory();
      byte[] byteArray0 = new byte[3];
      JsonParser jsonParser0 = jsonFactory0.createParser(byteArray0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer0.deserializeWithType(jsonParser0, defaultDeserializationContext_Impl0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      arrayNode0.addObject();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(arrayNode0);
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      StringArrayDeserializer stringArrayDeserializer2 = new StringArrayDeserializer(stringArrayDeserializer1);
      // Undeclared exception!
      try { 
        stringArrayDeserializer2._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      Float float0 = new Float((float) 3856);
      ArrayNode arrayNode1 = arrayNode0.insert(3396, float0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(arrayNode1);
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer1._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ArrayNode arrayNode1 = arrayNode0.add("");
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(arrayNode1);
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer1._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      ArrayNode arrayNode1 = arrayNode0.insert(3852, (Float) null);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(arrayNode1);
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        stringArrayDeserializer1._deserializeCustom(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      StringArrayDeserializer stringArrayDeserializer2 = new StringArrayDeserializer(stringArrayDeserializer1);
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(jsonParser0, jsonParser0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer2._deserializeCustom(jsonParserSequence0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      NullNode nullNode0 = NullNode.instance;
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(nullNode0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer0._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      JsonDeserializer<SimpleModule> jsonDeserializer0 = (JsonDeserializer<SimpleModule>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null, (Object) null, (Object) null, (Object) null, (Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer(jsonDeserializer0);
      // Undeclared exception!
      stringArrayDeserializer0._deserializeCustom(jsonParser0, defaultDeserializationContext_Impl0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.reader();
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      ArrayType arrayType0 = ArrayType.construct(simpleType0, objectMapper0, class0);
      ObjectReader objectReader1 = objectReader0.withType((Type) arrayType0);
      assertFalse(objectReader1.equals((Object)objectReader0));
  }
}