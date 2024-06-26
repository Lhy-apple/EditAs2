/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:24:47 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBase;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeBindings_ESTest extends TypeBindings_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeBindings.TypeParamStash typeBindings_TypeParamStash0 = new TypeBindings.TypeParamStash();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      JavaType[] javaTypeArray0 = typeBindings0.typeParameterArray();
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      TypeBindings typeBindings1 = typeBindings0.withUnboundVariable("");
      TypeBindings typeBindings2 = typeBindings1.withUnboundVariable("b.MOAKzLz");
      assertTrue(typeBindings2.equals((Object)typeBindings0));
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      typeBindings0.hashCode();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<MapLikeType> class0 = MapLikeType.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      Object object0 = typeBindings0.readResolve();
      assertSame(typeBindings0, object0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) simpleType0);
      TypeBindings typeBindings1 = (TypeBindings)typeBindings0.readResolve();
      assertEquals(1, typeBindings1.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<TypeBindings> class0 = TypeBindings.class;
      TypeBindings typeBindings0 = TypeBindings.create(class0, (List<JavaType>) null);
      assertEquals(0, typeBindings0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) stack0);
      assertEquals(0, typeBindings0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      Vector<JavaType> vector0 = new Vector<JavaType>();
      vector0.setSize(32);
      // Undeclared exception!
      try { 
        TypeBindings.create((Class<?>) class0, (List<JavaType>) vector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Stack with 32 type parameters: class expects 1
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      TypeBindings typeBindings0 = TypeBindings.create(class0, (JavaType[]) null);
      assertTrue(typeBindings0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      Class<TypeBindings> class0 = TypeBindings.class;
      // Undeclared exception!
      try { 
        TypeBindings.create((Class<?>) class0, (JavaType) simpleType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class com.fasterxml.jackson.databind.type.TypeBindings with 1 type parameter: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<String> class0 = String.class;
      JavaType[] javaTypeArray0 = new JavaType[2];
      // Undeclared exception!
      try { 
        TypeBindings.create(class0, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.lang.String with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Stack> class0 = Stack.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      Class<ResolvedRecursiveType> class1 = ResolvedRecursiveType.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class1, (JavaType) collectionType0);
      assertTrue(typeBindings0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      // Undeclared exception!
      try { 
        TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) mapType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.HashMap with 1 type parameter: class expects 2
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      // Undeclared exception!
      try { 
        TypeBindings.createIfNeeded(class0, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.HashMap with 1 type parameter: class expects 2
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      // Undeclared exception!
      try { 
        TypeBindings.createIfNeeded(class0, (JavaType[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Stack with 0 type parameters: class expects 1
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class1 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType(class1, class0, class1);
      assertEquals(2, mapType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class1 = HashMap.class;
      typeFactory0.constructMapType((Class<? extends Map>) class1, (JavaType) resolvedRecursiveType0, (JavaType) resolvedRecursiveType0);
      assertTrue(typeBindings0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      TypeFactory typeFactory0 = TypeFactory.instance;
      resolvedRecursiveType0.setReference(resolvedRecursiveType0);
      Class<HashMap> class1 = HashMap.class;
      typeFactory0.constructMapType((Class<? extends Map>) class1, (JavaType) resolvedRecursiveType0, (JavaType) resolvedRecursiveType0);
      assertTrue(typeBindings0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      String string0 = typeBindings0.getBoundName((-50));
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      javaTypeArray0[0] = (JavaType) simpleType0;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaTypeArray0);
      String string0 = typeBindings0.getBoundName(0);
      assertNotNull(string0);
      assertEquals(1, typeBindings0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      String string0 = typeBindings0.getBoundName(0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      JavaType javaType0 = simpleType0.containedTypeOrUnknown((-667));
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Stack> class0 = Stack.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      JavaType javaType0 = collectionType0.containedTypeOrUnknown(0);
      assertFalse(javaType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      JavaType javaType0 = typeBindings0.getBoundType(2);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      List<JavaType> list0 = typeBindings0.getTypeParameters();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      TypeBindings typeBindings1 = typeBindings0.withUnboundVariable("");
      boolean boolean0 = typeBindings1.hasUnbound("@c");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      TypeBindings typeBindings1 = typeBindings0.withUnboundVariable("");
      boolean boolean0 = typeBindings1.hasUnbound("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      String string0 = typeBindings0.toString();
      assertEquals("<>", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      String string0 = typeBindings0.toString();
      assertEquals("<Ljava/lang/Class;,Ljava/lang/Class;>", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      boolean boolean0 = typeBindings0.equals(stack0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      boolean boolean0 = typeBindings0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Stack> class0 = Stack.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) collectionType0);
      TypeBindings typeBindings1 = TypeBindings.emptyBindings();
      boolean boolean0 = typeBindings0.equals(typeBindings1);
      assertFalse(typeBindings0.isEmpty());
      assertTrue(typeBindings1.isEmpty());
      assertFalse(typeBindings1.equals((Object)typeBindings0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<String> class0 = String.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings1 = TypeBindings.create(class0, javaTypeArray0);
      boolean boolean0 = typeBindings1.equals(typeBindings0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Class<MapLikeType> class0 = MapLikeType.class;
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectReader objectReader0 = objectMapper0.reader(jsonNodeFactory0);
      JavaType javaType0 = TypeBase._bogusSuperClass(class0);
      ObjectReader objectReader1 = objectReader0.forType(javaType0);
      assertNotSame(objectReader0, objectReader1);
  }
}
