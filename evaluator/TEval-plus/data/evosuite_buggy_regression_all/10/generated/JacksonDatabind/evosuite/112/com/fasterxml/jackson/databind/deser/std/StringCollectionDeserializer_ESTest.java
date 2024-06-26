/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:44:29 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.filter.FilteringParserDelegate;
import com.fasterxml.jackson.core.filter.TokenFilter;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.NullValueProvider;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer;
import com.fasterxml.jackson.databind.ext.DOMDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.DoubleNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringCollectionDeserializer_ESTest extends StringCollectionDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Boolean boolean0 = Boolean.TRUE;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      try { 
        stringCollectionDeserializer0.getEmptyValue((DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Cannot create empty instance of [simple type, class java.lang.Object], no default Creator
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<HashMap> class1 = HashMap.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ArrayType arrayType0 = ArrayType.construct((JavaType) simpleType0, typeBindings0);
      JsonDeserializer<SimpleObjectIdResolver> jsonDeserializer0 = (JsonDeserializer<SimpleObjectIdResolver>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class1);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(arrayType0, jsonDeserializer0, valueInstantiator_Base0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserializeWithType(jsonParser0, defaultDeserializationContext_Impl0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      Boolean boolean0 = Boolean.FALSE;
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      assertNotSame(stringCollectionDeserializer1, stringCollectionDeserializer0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, jsonDeserializer0, (ValueInstantiator) null);
      JsonDeserializer<String> jsonDeserializer1 = stringCollectionDeserializer0._valueDeserializer;
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved(jsonDeserializer1, jsonDeserializer1, jsonDeserializer1, (Boolean) null);
      assertFalse(stringCollectionDeserializer1.isCachable());
      assertNotSame(stringCollectionDeserializer1, stringCollectionDeserializer0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(mapType0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(mapType0, jsonDeserializer0, valueInstantiator_Base0);
      boolean boolean0 = stringCollectionDeserializer0.isCachable();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(jsonDeserializer0).toString();
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, jsonDeserializer0, valueInstantiator_Base0);
      JsonDeserializer<Object> jsonDeserializer1 = stringCollectionDeserializer0.getContentDeserializer();
      DOMDeserializer.NodeDeserializer dOMDeserializer_NodeDeserializer0 = new DOMDeserializer.NodeDeserializer();
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved(jsonDeserializer1, (JsonDeserializer<?>) null, dOMDeserializer_NodeDeserializer0, (Boolean) null);
      boolean boolean0 = stringCollectionDeserializer1.isCachable();
      assertFalse(boolean0);
      assertFalse(stringCollectionDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JsonDeserializer<InputStream> jsonDeserializer0 = (JsonDeserializer<InputStream>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, jsonDeserializer0, valueInstantiator_Base0);
      StringCollectionDeserializer stringCollectionDeserializer1 = stringCollectionDeserializer0.withResolved((JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, (Boolean) null);
      boolean boolean0 = stringCollectionDeserializer1.isCachable();
      assertTrue(boolean0);
      assertFalse(stringCollectionDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (JsonDeserializer<?>) null, (ValueInstantiator) null);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.createContextual((DeserializationContext) null, beanProperty_Bogus0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.createContextual((DeserializationContext) null, beanProperty_Bogus0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      JsonDeserializer<DoubleNode> jsonDeserializer0 = (JsonDeserializer<DoubleNode>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, jsonDeserializer0, valueInstantiator_Base0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer(simpleType0, stringCollectionDeserializer0, valueInstantiator_Base0);
      assertFalse(stringCollectionDeserializer1.isCachable());
      
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonDeserializer<?> jsonDeserializer1 = stringCollectionDeserializer1.createContextual(defaultDeserializationContext_Impl0, beanProperty_Bogus0);
      assertTrue(jsonDeserializer1.isCachable());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      Stack<String> stack0 = new Stack<String>();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Boolean boolean0 = Boolean.TRUE;
      StringCollectionDeserializer stringCollectionDeserializer1 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, stringCollectionDeserializer0, stringCollectionDeserializer0, stringCollectionDeserializer0, boolean0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer1.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) stack0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Integer) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      Boolean boolean0 = new Boolean("atlKX2 Q");
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, valueInstantiator_Base0, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, boolean0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize((JsonParser) null, (DeserializationContext) defaultDeserializationContext_Impl0);
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
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(jsonParser0, tokenFilter0, true, true);
      filteringParserDelegate0.nextBooleanValue();
      Class<DoubleNode> class0 = DoubleNode.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (JsonDeserializer<?>) null, valueInstantiator_Base0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Collection<String> collection0 = stringCollectionDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) linkedHashSet0);
      try { 
        stringCollectionDeserializer0.deserialize((JsonParser) filteringParserDelegate0, (DeserializationContext) defaultDeserializationContext_Impl0, collection0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // (was java.lang.NullPointerException) (through reference chain: java.util.LinkedHashSet[0])
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Boolean boolean0 = Boolean.TRUE;
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      JsonParser jsonParser0 = arrayNode0.traverse();
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(jsonParser0, tokenFilter0, true, true);
      filteringParserDelegate0.nextBooleanValue();
      Class<DoubleNode> class0 = DoubleNode.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      JsonDeserializer<ObjectIdResolver> jsonDeserializer0 = (JsonDeserializer<ObjectIdResolver>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, valueInstantiator_Base0, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, boolean0);
      List<String> list0 = arrayNode0.findValuesAsText("");
      Collection<String> collection0 = stringCollectionDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) list0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize((JsonParser) filteringParserDelegate0, (DeserializationContext) defaultDeserializationContext_Impl0, collection0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.AbstractList", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      Stack<String> stack0 = new Stack<String>();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer((JavaType) null, (JsonDeserializer<?>) null, (ValueInstantiator) null);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) stack0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      Stack<String> stack0 = new Stack<String>();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Boolean boolean0 = Boolean.valueOf("JSON");
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) stack0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createNonBlockingByteArrayParser();
      Stack<String> stack0 = new Stack<String>();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Boolean boolean0 = Boolean.TRUE;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      StringCollectionDeserializer stringCollectionDeserializer0 = new StringCollectionDeserializer(simpleType0, valueInstantiator_Base0, (JsonDeserializer<?>) null, (JsonDeserializer<?>) null, (NullValueProvider) null, boolean0);
      // Undeclared exception!
      try { 
        stringCollectionDeserializer0.deserialize(jsonParser0, (DeserializationContext) defaultDeserializationContext_Impl0, (Collection<String>) stack0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }
}
