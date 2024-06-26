/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:00:45 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.node.ShortNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;
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
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType[] javaTypeArray0 = typeBindings0.typeParameterArray();
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      typeBindings0.hashCode();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(defaultSerializerProvider_Impl0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) stack0);
      Object object0 = typeBindings0.readResolve();
      assertNotSame(object0, typeBindings0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (JavaType) simpleType0);
      TypeBindings typeBindings1 = (TypeBindings)typeBindings0.readResolve();
      assertFalse(typeBindings1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.create(class0, (List<JavaType>) null);
      assertTrue(typeBindings0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      LinkedList<JavaType> linkedList0 = new LinkedList<JavaType>();
      linkedList0.add((JavaType) simpleType0);
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) linkedList0);
      assertFalse(typeBindings0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      // Undeclared exception!
      try { 
        TypeBindings.create(class0, (JavaType[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Stack with 0 type parameters: class expects 1
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        TypeBindings.create(class0, (JavaType) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.lang.Object with 1 type parameter: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<MapType> class0 = MapType.class;
      // Undeclared exception!
      try { 
        TypeBindings.create(class0, (JavaType) null, (JavaType) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class com.fasterxml.jackson.databind.type.MapType with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class1 = HashMap.class;
      Class<ShortNode> class2 = ShortNode.class;
      MapType mapType0 = typeFactory0.constructMapType(class1, class0, class2);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(mapType0, mapType0);
      // Undeclared exception!
      try { 
        TypeBindings.createIfNeeded((Class<?>) class1, (JavaType) referenceType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.HashMap with 1 type parameter: class expects 2
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) simpleType0);
      assertEquals(0, typeBindings0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) simpleType0);
      String string0 = typeBindings0.getBoundName(0);
      assertNotNull(string0);
      assertEquals(1, typeBindings0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      // Undeclared exception!
      try { 
        TypeBindings.createIfNeeded(class0, javaTypeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
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
  public void test16()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (JavaType) simpleType0);
      TypeBindings typeBindings1 = typeBindings0.withUnboundVariable("6@)%");
      TypeBindings typeBindings2 = typeBindings1.withUnboundVariable("6@)%");
      assertNotSame(typeBindings0, typeBindings2);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Stack<JavaType> stack0 = new Stack<JavaType>();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(stack0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, (JavaType[]) null);
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      Class<Stack> class1 = Stack.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      resolvedRecursiveType0.setReference(simpleType0);
      CollectionType collectionType0 = typeFactory0.constructCollectionType((Class<? extends Collection>) class1, (JavaType) resolvedRecursiveType0);
      assertTrue(typeBindings0.isEmpty());
      assertEquals(1, collectionType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      String string0 = typeBindings0.getBoundName((-2174));
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      String string0 = typeBindings0.getBoundName(0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType javaType0 = typeBindings0.getBoundType((-5));
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType javaType0 = typeBindings0.getBoundType(0);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      List<JavaType> list0 = typeBindings0.getTypeParameters();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) stack0);
      TypeBindings typeBindings1 = typeBindings0.withUnboundVariable("3iO'1rF^e");
      boolean boolean0 = typeBindings1.hasUnbound("3iO'1rF^e");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      TypeBindings typeBindings1 = typeBindings0.withUnboundVariable("N;_ab$1@dKn157yW*c");
      boolean boolean0 = typeBindings1.hasUnbound("LYk`=G1Ee6D\"hdn-[Nk");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<Stack> class0 = Stack.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (JavaType) simpleType0);
      String string0 = typeBindings0.toString();
      assertEquals("<Ljava/lang/Class;>", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      String string0 = typeBindings0.toString();
      assertEquals("<>", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      boolean boolean0 = typeBindings0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      Stack<JavaType> stack0 = new Stack<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) stack0);
      boolean boolean0 = typeBindings0.equals(class0);
      assertFalse(boolean0);
  }
}
