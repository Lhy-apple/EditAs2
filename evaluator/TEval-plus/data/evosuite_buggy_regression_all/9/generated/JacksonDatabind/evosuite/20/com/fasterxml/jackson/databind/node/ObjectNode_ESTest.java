/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:41:46 GMT 2023
 */

package com.fasterxml.jackson.databind.node;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonPointer;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.cfg.ContextAttributes;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeSerializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.FloatNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.JsonNodeType;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ValueNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.ByteArrayOutputStream;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ObjectNode_ESTest extends ObjectNode_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      ObjectNode objectNode1 = objectNode0.putNull("com.fasterxml.jackson.databind.deser.std.BaseNodeDeserializer");
      assertFalse(objectNode1.booleanValue());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      JsonNode jsonNode0 = objectNode0.putAll(objectNode0);
      assertNull(jsonNode0.textValue());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper((JsonFactory) null);
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      Iterator<String> iterator0 = objectNode0.fieldNames();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      objectNode0.hashCode();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      // Undeclared exception!
      try { 
        objectNode0._at((JsonPointer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.node.ObjectNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put((String) null, (short)823);
      assertFalse(objectNode1.isBigDecimal());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.removeAll();
      assertFalse(objectNode1.isDouble());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      Consumer<Object> consumer0 = (Consumer<Object>) mock(Consumer.class, new ViolatedAssumptionAnswer());
      objectNode0.forEach(consumer0);
      assertFalse(objectNode0.isFloat());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      JsonNode jsonNode0 = objectNode0.without("C?ch7Gpj");
      assertFalse(jsonNode0.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      JsonToken jsonToken0 = objectNode0.asToken();
      assertEquals("{", jsonToken0.asString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      boolean boolean0 = objectNode0.hasNonNull("{xnF5W>(*7iv38+2&-");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      String[] stringArray0 = new String[5];
      ObjectNode objectNode1 = objectNode0.retain(stringArray0);
      assertNull(objectNode1.textValue());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonParser jsonParser0 = objectNode0.traverse();
      assertFalse(jsonParser0.isExpectedStartArrayToken());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      // Undeclared exception!
      try { 
        objectNode0.remove((Collection<String>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Objects", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      JsonNode jsonNode0 = objectNode0.remove("");
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      boolean boolean0 = objectNode0.has((-3450));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      ObjectNode objectNode1 = objectNode0.put("com.fasterxml.jackson.databind.deser.std.BaseNodeDeserializer", 0L);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      NullNode nullNode0 = jsonNodeFactory0.nullNode();
      IOContext iOContext0 = new IOContext(bufferRecycler0, nullNode0, false);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ContextAttributes contextAttributes0 = ContextAttributes.Impl.getEmpty();
      ObjectReader objectReader0 = objectMapper0.reader(contextAttributes0);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, (-1611004515), objectReader0, byteArrayBuilder0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, typeFactory0);
      PropertyName propertyName0 = PropertyName.construct("com.fasterxml.jackson.databind.deser.std.BaseNodeDeserializer", "~2*");
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, classNameIdResolver0, "com.fasterxml.jackson.databind.deser.std.BaseNodeDeserializer", false, class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, asPropertyTypeDeserializer0, annotationMap0, annotatedParameter0, 360, objectNode1, propertyMetadata0);
      AsPropertyTypeSerializer asPropertyTypeSerializer0 = new AsPropertyTypeSerializer(classNameIdResolver0, creatorProperty0, "6J");
      objectNode0.serializeWithType(uTF8JsonGenerator0, serializerProvider0, asPropertyTypeSerializer0);
      assertEquals(JsonNodeType.OBJECT, objectNode0.getNodeType());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.putObject("C?ch7Gpj");
      boolean boolean0 = objectNode1.equals(objectNode0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("8tpDg_I8", true);
      assertEquals("", objectNode1.asText());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ArrayList<String> arrayList0 = new ArrayList<String>();
      ObjectNode objectNode1 = objectNode0.without((Collection<String>) arrayList0);
      assertSame(objectNode0, objectNode1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      JsonNode jsonNode0 = objectNode0.path(0);
      assertFalse(jsonNode0.isDouble());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ArrayNode arrayNode0 = objectNode0.putArray("om$YD1");
      ObjectNode objectNode1 = arrayNode0.addObject();
      ObjectNode objectNode2 = objectNode1.with("om$YD1");
      JsonNode jsonNode0 = objectNode1.findValue("' has value that is not of type ObjectNode (but ");
      assertNull(jsonNode0);
      assertNotSame(objectNode1, objectNode2);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.putPOJO("declaringClass", "declaringClass");
      assertFalse(objectNode1.isShort());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("C?ch7Gpj", "C?ch7Gpj");
      ObjectNode objectNode2 = objectNode1.deepCopy();
      assertEquals(1, objectNode2.size());
      assertNotSame(objectNode2, objectNode0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      JsonNode jsonNode0 = objectNode0.path("6k?r;n@f8{Rzh\"]Hn");
      assertFalse(jsonNode0.isDouble());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      Double double0 = new Double(1.0);
      ObjectNode objectNode1 = objectNode0.put("", double0);
      JsonNode jsonNode0 = objectNode1.path("");
      assertEquals(1.0F, jsonNode0.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("8tpaD5_I8", (-271));
      // Undeclared exception!
      try { 
        objectNode1.with("8tpaD5_I8");
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Property '8tpaD5_I8' has value that is not of type ObjectNode (but com.fasterxml.jackson.databind.node.IntNode)
         //
         verifyException("com.fasterxml.jackson.databind.node.ObjectNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.replace("X1a", objectNode0);
      ObjectNode objectNode1 = objectNode0.with("X1a");
      assertFalse(objectNode1.isShort());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      objectNode0.withArray("Ub~H7*");
      ArrayNode arrayNode1 = objectNode0.withArray("Ub~H7*");
      assertFalse(arrayNode1.isBigDecimal());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      Double double0 = new Double(683.5400774185086);
      ObjectNode objectNode1 = objectNode0.put("t)JXh;*KF]W6M", double0);
      // Undeclared exception!
      try { 
        objectNode1.withArray("t)JXh;*KF]W6M");
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Property 't)JXh;*KF]W6M' has value that is not of type ArrayNode (but com.fasterxml.jackson.databind.node.DoubleNode)
         //
         verifyException("com.fasterxml.jackson.databind.node.ObjectNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      JsonNode jsonNode0 = objectNode0.findValue("}w");
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ArrayNode arrayNode0 = objectNode0.putArray("om$YD1");
      ObjectNode objectNode1 = arrayNode0.addObject();
      ObjectNode objectNode2 = objectNode1.with("om$YD1");
      objectNode2.put("' has value that is not of type ObjectNode (but ", (Boolean) null);
      objectNode1.findValue("' has value that is not of type ObjectNode (but ");
      assertEquals(1, objectNode1.size());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("tp6D_I8", "tp6D_I8");
      List<JsonNode> list0 = objectNode1.findValues("Unexpected end-of-input when trying to deserialize a ");
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("8tpDg_I8", "8tpDg_I8");
      List<JsonNode> list0 = objectNode1.findValues("8tpDg_I8");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Float float0 = new Float(1419.0555437942);
      objectNode0.put("8tp/g_I8", float0);
      ArrayList<JsonNode> arrayList0 = new ArrayList<JsonNode>();
      objectNode0.findValues("8tp/g_I8", (List<JsonNode>) arrayList0);
      assertEquals(1, arrayList0.size());
      assertFalse(arrayList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("8tpD_I", "8tpD_I");
      List<String> list0 = objectNode1.findValuesAsText("8tpD_I");
      assertTrue(list0.contains("8tpD_I"));
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("8tp/g_I8", "8tp/g_I8");
      List<String> list0 = objectNode1.findValuesAsText("com.fasterxml.jackson.annotation.PropertyAccessor");
      // Undeclared exception!
      try { 
        objectNode0.findValuesAsText("8tp/g_I8", list0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.AbstractList", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      objectNode0.withArray("Ub~H7*");
      ObjectNode objectNode1 = objectNode0.findParent(" bytes (out of ");
      assertNull(objectNode1);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      objectNode0.put(" bytes (out of ", 1200.0);
      ObjectNode objectNode1 = objectNode0.findParent(" bytes (out of ");
      assertEquals(JsonToken.START_OBJECT, objectNode1.asToken());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ObjectNode objectNode1 = objectNode0.with("");
      BigDecimal bigDecimal0 = BigDecimal.ZERO;
      objectNode1.put("': expected '", bigDecimal0);
      objectNode0.findParent("': expected '");
      assertEquals(1, objectNode0.size());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("C?ch7Gpj", "C?ch7Gpj");
      ArrayList<JsonNode> arrayList0 = new ArrayList<JsonNode>();
      objectNode1.findParents("C?ch7Gpj", (List<JsonNode>) arrayList0);
      assertEquals(1, arrayList0.size());
      assertFalse(arrayList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      ObjectNode objectNode0 = arrayNode0.insertObject(4395);
      Short short0 = new Short((short)2047);
      objectNode0.put("", short0);
      List<JsonNode> list0 = objectNode0.findParents("`,Bf['t.r>0|B4X]L", (List<JsonNode>) null);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      ObjectNode objectNode0 = arrayNode0.insertObject(4395);
      ObjectNode objectNode1 = objectNode0.putObject("`,Bf['t.r>0|B4X]L");
      List<JsonNode> list0 = objectNode0.findParents("`,Bf['t.r>0|B4X]L", (List<JsonNode>) null);
      assertNotSame(objectNode0, objectNode1);
      assertNotNull(list0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Integer integer0 = new Integer((-2296));
      ValueNode valueNode0 = jsonNodeFactory0.numberNode(integer0);
      ObjectNode objectNode1 = objectNode0.with("*A3*");
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, valueNode0, false);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(65535);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 9, objectMapper0, byteArrayOutputStream0);
      objectNode0.serialize(uTF8JsonGenerator0, (SerializerProvider) null);
      assertNotSame(objectNode0, objectNode1);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      NullNode nullNode0 = jsonNodeFactory0.nullNode();
      IOContext iOContext0 = new IOContext(bufferRecycler0, nullNode0, false);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ContextAttributes contextAttributes0 = ContextAttributes.Impl.getEmpty();
      ObjectReader objectReader0 = objectMapper0.reader(contextAttributes0);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, (-1611004515), objectReader0, byteArrayBuilder0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, typeFactory0);
      PropertyName propertyName0 = PropertyName.construct("com.fasterxml.jackson.databind.deser.std.BaseNodeDeserializer", "~2*");
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(simpleType0, classNameIdResolver0, "com.fasterxml.jackson.databind.deser.std.BaseNodeDeserializer", false, class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, simpleType0, annotationMap0, 0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, asPropertyTypeDeserializer0, annotationMap0, annotatedParameter0, 360, objectNode0, propertyMetadata0);
      AsPropertyTypeSerializer asPropertyTypeSerializer0 = new AsPropertyTypeSerializer(classNameIdResolver0, creatorProperty0, "6J");
      objectNode0.serializeWithType(uTF8JsonGenerator0, serializerProvider0, asPropertyTypeSerializer0);
      assertFalse(uTF8JsonGenerator0.canWriteObjectId());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      JsonNode jsonNode0 = objectNode0.set("JSON", objectNode0);
      assertNull(jsonNode0.textValue());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      JsonNode jsonNode0 = objectNode0.set("8tp/Dg_I8", (JsonNode) null);
      assertFalse(jsonNode0.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      HashMap<String, FloatNode> hashMap0 = new HashMap<String, FloatNode>();
      FloatNode floatNode0 = FloatNode.valueOf(0.0F);
      hashMap0.put("2", floatNode0);
      objectNode0.putAll((Map<String, ? extends JsonNode>) hashMap0);
      assertEquals(1, objectNode0.size());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      hashMap0.put(")", (JsonNode) null);
      JsonNode jsonNode0 = objectNode0.setAll((Map<String, ? extends JsonNode>) hashMap0);
      assertEquals("", jsonNode0.asText());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      JsonNode jsonNode0 = objectNode0.replace("a|WQ.6pu1HtS", (JsonNode) null);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ArrayNode arrayNode0 = objectNode0.arrayNode();
      JsonNode jsonNode0 = objectNode0.put("", (JsonNode) arrayNode0);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      JsonNode jsonNode0 = objectNode0.put("Incompatible narrowing operation: trying to narrow ", (JsonNode) null);
      assertNull(jsonNode0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put((String) null, (Short) null);
      assertFalse(objectNode1.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Integer integer0 = new Integer((-2034458190));
      ObjectNode objectNode1 = objectNode0.put((String) null, integer0);
      assertFalse(objectNode1.isLong());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("B}*", (Integer) null);
      assertFalse(objectNode1.isShort());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      Long long0 = new Long(249L);
      ObjectNode objectNode1 = objectNode0.put("8tp/Dg_I8", long0);
      assertEquals("", objectNode1.asText());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put((String) null, (Long) null);
      assertSame(objectNode1, objectNode0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("declaringClass", (Float) null);
      assertFalse(objectNode1.isBigInteger());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("-*bq#ZM?^u*7pH&2", (Double) null);
      assertFalse(objectNode1.isLong());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put((String) null, (BigDecimal) null);
      assertFalse(objectNode1.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put((String) null, (String) null);
      assertFalse(objectNode1.isFloat());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      Boolean boolean0 = Boolean.FALSE;
      ObjectNode objectNode1 = objectNode0.put("HG.DD}n0%6>", boolean0);
      assertEquals(JsonNodeType.OBJECT, objectNode1.getNodeType());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.addObject();
      byte[] byteArray0 = new byte[2];
      ObjectNode objectNode1 = objectNode0.put("[()\"\n['9x$2*ZP", byteArray0);
      assertFalse(objectNode1.isFloat());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ObjectNode objectNode1 = objectNode0.put("8tp/g_I8", (byte[]) null);
      assertSame(objectNode1, objectNode0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      boolean boolean0 = objectNode0.equals(objectNode0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      boolean boolean0 = objectNode0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      NullNode nullNode0 = objectNode0.nullNode();
      boolean boolean0 = objectNode0.equals(nullNode0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("C?ch7Gpj", "C?ch7Gpj");
      String string0 = objectNode1.toString();
      assertEquals("{\"C?ch7Gpj\":\"C?ch7Gpj\"}", string0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      ObjectNode objectNode1 = objectNode0.put("Failed to parse Date value '", 0.0F);
      BigDecimal bigDecimal0 = new BigDecimal(0);
      ObjectNode objectNode2 = objectNode1.put("xn7!KaZykn1}G\"T\"@", bigDecimal0);
      String string0 = objectNode2.toString();
      assertEquals("{\"Failed to parse Date value '\":0.0,\"xn7!KaZykn1}G\\\"T\\\"@\":0}", string0);
  }
}