/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:05:14 GMT 2023
 */

package org.apache.commons.math.geometry.euclidean.threed;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.geometry.euclidean.threed.Rotation;
import org.apache.commons.math.geometry.euclidean.threed.RotationOrder;
import org.apache.commons.math.geometry.euclidean.threed.Vector3D;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Rotation_ESTest extends Rotation_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      Rotation.distance(rotation0, rotation0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(0.0, 0.0);
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      Rotation rotation1 = rotation0.revert();
      assertEquals(Double.NaN, rotation1.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(0.0, 0.0);
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      double double0 = rotation0.getQ3();
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      double double0 = rotation0.getQ1();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      Rotation rotation0 = new Rotation(rotationOrder0, 1058.422865854077, 1058.422865854077, 0.0);
      double double0 = rotation0.getQ2();
      assertEquals((-0.14482369051079141), rotation0.getQ3(), 0.01);
      assertEquals((-0.9785667128696212), rotation0.getQ1(), 0.01);
      assertEquals(0.02143328713037875, rotation0.getQ0(), 0.01);
      assertEquals((-0.14482369051079141), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      double double0 = rotation0.getQ0();
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {-0.0, 0.0, -0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = null;
      try {
        rotation1 = new Rotation(doubleArray0, 3.141592653589793);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // unable to orthogonalize matrix in 10 iterations
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Rotation rotation0 = new Rotation((-176.0), (-176.0), (-176.0), 1849.195601782, true);
      double double0 = rotation0.getAngle();
      assertEquals(2.953497394426802, double0, 0.01);
      assertEquals(0.9866829638560306, rotation0.getQ3(), 0.01);
      assertEquals((-0.09390904968155638), rotation0.getQ0(), 0.01);
      assertEquals((-0.09390904968155638), rotation0.getQ1(), 0.01);
      assertEquals((-0.09390904968155638), rotation0.getQ2(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, (-1833.155359803746));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero norm for rotation axis
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double[][] doubleArray0 = new double[0][2];
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(doubleArray0, 0.038539716627904085);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[][] doubleArray0 = new double[3][0];
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(doubleArray0, 0.22874110367859227);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // a 3x0 matrix cannot be a rotation matrix
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      double[] doubleArray0 = rotation0.IDENTITY.getAngles(rotationOrder0);
      double[][] doubleArray1 = new double[3][9];
      doubleArray1[0] = doubleArray0;
      Rotation rotation1 = null;
      try {
        rotation1 = new Rotation(doubleArray1, 0.0);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // a 3x3 matrix cannot be a rotation matrix
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      double[] doubleArray0 = rotation0.IDENTITY.getAngles(rotationOrder0);
      double[][] doubleArray1 = new double[3][9];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      Rotation rotation1 = null;
      try {
        rotation1 = new Rotation(doubleArray1, 0.0);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // a 3x3 matrix cannot be a rotation matrix
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Vector3D vector3D1 = Vector3D.MINUS_J;
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D1, vector3D0);
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = new Rotation(doubleArray0, 3.141592653589793);
      assertEquals(0.0, rotation1.getQ1(), 0.01);
      assertEquals(0.7071067811865475, rotation1.getQ3(), 0.01);
      assertEquals(0.7071067811865475, rotation0.getQ2(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      Vector3D vector3D1 = rotationOrder0.getA3();
      Vector3D vector3D2 = new Vector3D(2372.8234, vector3D0, (-1044.4791972), vector3D0);
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D2, vector3D1);
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = null;
      try {
        rotation1 = new Rotation(doubleArray0, 2013.1419586919224);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // the closest orthogonal matrix has a negative determinant -1
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = new Rotation(doubleArray0, 2.0);
      assertEquals(1.0, rotation1.getQ0(), 0.01);
      assertEquals(0.0, rotation0.getQ2(), 0.01);
      assertEquals(1.0, rotation0.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(0.0, 0.0);
      RotationOrder rotationOrder0 = RotationOrder.ZYZ;
      Vector3D vector3D1 = rotationOrder0.getA1();
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D1, vector3D0);
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = new Rotation(doubleArray0, 0.14491500684942465);
      assertEquals(0.7071067811865475, rotation1.getQ3(), 0.01);
      assertEquals(-0.0, rotation1.getQ0(), 0.01);
      assertEquals(0.0, rotation0.getQ2(), 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Rotation rotation0 = new Rotation(0.0, 681.8067, (-707.3835), (-2807.0), true);
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = new Rotation(doubleArray0, 0.9278720362640083);
      assertEquals((-0.943856019809017), rotation0.getQ3(), 0.01);
      assertEquals((-0.23785827388264047), rotation0.getQ2(), 0.01);
      assertEquals(0.22925805420061293, rotation0.getQ1(), 0.01);
      assertEquals((-0.22925805420061288), rotation1.getQ1(), 0.01);
      assertEquals(0.0, rotation0.getQ0(), 0.01);
      assertEquals(0.0, rotation1.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_J;
      Vector3D vector3D1 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D1, vector3D1, vector3D1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Vector3D vector3D1 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0, vector3D1, vector3D1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Vector3D vector3D1 = Vector3D.crossProduct(vector3D0, vector3D0);
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Vector3D vector3D1 = new Vector3D(0.1, 0.17460232679752388);
      Vector3D vector3D2 = Vector3D.MINUS_J;
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D0, vector3D2);
      assertEquals(0.5037147673659403, rotation0.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Vector3D vector3D1 = new Vector3D(0.2375878600396614, vector3D0, (-0.27376775920738433), vector3D0);
      Rotation rotation0 = new Rotation(vector3D0, vector3D1);
      RotationOrder rotationOrder0 = RotationOrder.YZY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      Vector3D vector3D1 = rotation0.getAxis();
      assertEquals(Double.NaN, vector3D1.getNormInf(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      Vector3D vector3D1 = rotation0.IDENTITY.getAxis();
      assertEquals(1.0, vector3D1.getX(), 0.01);
      assertEquals(Double.NaN, rotation0.getQ1(), 0.01);
      assertEquals(1.0, vector3D1.getNorm1(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Rotation rotation0 = new Rotation(458.516132281796, 0.0, 458.516132281796, (-5279.26), false);
      Rotation rotation1 = new Rotation(0.0, 3649.655790112848, 3649.655790112848, 0.0, false);
      Rotation rotation2 = rotation0.applyInverseTo(rotation1);
      Vector3D vector3D0 = rotation2.getAxis();
      assertEquals((-0.7642093149952004), vector3D0.getX(), 0.01);
      assertEquals(0.06106935706220593, vector3D0.getZ(), 0.01);
      assertEquals((-1673426.0570424052), rotation2.getQ0(), 0.01);
      assertEquals(0.6420706008707885, vector3D0.getY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      double double0 = rotation0.getAngle();
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      double double0 = rotation0.getAngle();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(0.0, 0.0);
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XYZ;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XYZ;
      Rotation rotation0 = new Rotation((-0.8542460820732686), 0.14491500684942465, (-0.8542460820732686), (-0.8542460820732686), false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(0.0, 0.0);
      RotationOrder rotationOrder0 = RotationOrder.ZYZ;
      Vector3D vector3D1 = rotationOrder0.getA1();
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D1, vector3D0);
      RotationOrder rotationOrder1 = RotationOrder.XYZ;
      try { 
        rotation0.getAngles(rotationOrder1);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      Rotation rotation0 = new Rotation(81.0, (-851.8828955), 1629.940613537688, (-1890.976794759637), false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Rotation rotation0 = new Rotation((-1529.1109865168964), (-1529.1109865168964), (-1529.1109865168964), (-1529.1109865168964), true);
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Rotation rotation0 = new Rotation(1583.0969892, (-1421.829132964), 1583.0969892, (-2370.8615), false);
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      Rotation rotation0 = new Rotation(0.9449712005235171, 0.9449712005235171, 0.9449712005235171, 1.9002926521083813E7, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      RotationOrder rotationOrder0 = RotationOrder.ZYZ;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      Rotation rotation0 = new Rotation(0.5758187523046834, (-7.366802873448174), 0.5758187523046834, 3066.3469, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      Rotation rotation0 = new Rotation((-148.7), (-148.7), (-2.356194490192345), 0.5, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.ZXY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      Vector3D vector3D1 = rotationOrder0.getA2();
      Vector3D vector3D2 = Vector3D.PLUS_J;
      Rotation rotation0 = new Rotation(vector3D1, vector3D0, vector3D2, vector3D0);
      RotationOrder rotationOrder1 = RotationOrder.ZXY;
      try { 
        rotation0.getAngles(rotationOrder1);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXY;
      Rotation rotation0 = new Rotation((-291.419607), (-291.419607), (-1880.0), (-419.725244285), false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.NaN;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.ZYX;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Rotation rotation0 = new Rotation(5.811761241132772E13, 5.811761241132772E13, 0.1111111111111111, (-875.52525), false);
      RotationOrder rotationOrder0 = RotationOrder.ZYX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Rotation rotation0 = new Rotation(777.66887789, 1404.623031, 1404.623031, 777.66887789, false);
      RotationOrder rotationOrder0 = RotationOrder.ZYX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XYX;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Vector3D vector3D1 = Vector3D.MINUS_J;
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D1, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XYX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XYX;
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Vector3D vector3D1 = rotationOrder0.getA3();
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D1, vector3D0);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XZX;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XZX;
      Rotation rotation0 = new Rotation((-2.2250738585072014E-308), (-2.2250738585072014E-308), 241.377026, 241.377026, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.XZX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(0.0, 0.0);
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YXY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(0.0, 0.0);
      RotationOrder rotationOrder0 = RotationOrder.ZYZ;
      Vector3D vector3D1 = rotationOrder0.getA1();
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D1, vector3D0);
      RotationOrder rotationOrder1 = RotationOrder.YXY;
      try { 
        rotation0.getAngles(rotationOrder1);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YXY;
      Rotation rotation0 = Rotation.IDENTITY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YZY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YZY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(0.0, 0.0);
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Rotation rotation0 = new Rotation(0.0, (-3058.785599465427), 1952.3691582432702, 0.0, true);
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      Rotation rotation0 = Rotation.IDENTITY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZYZ;
      Rotation rotation0 = new Rotation(rotationOrder0, 3.141592653589793, 3.141592653589793, 916.414746233603);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZYZ;
      Rotation rotation0 = Rotation.IDENTITY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }
}
